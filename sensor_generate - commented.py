# ============================= size- & prevalence‑controlled generator =========================
# The calibration routines DO NOT modify the original generator; instead they wrap it so we can
# request an exact number of examples (`num_examples`) and, optionally, a target fraction of
# positives in the TRAIN split (`tr_frac`).  Both are achieved by iteratively choosing
# the number of sinusoid cycles and tuning the (thresh_a, thresh_b) pair.
#
# -----------------------------------------------------------------------------------------------
# CONTEXT / CONCEPTUAL TIE-IN (to the attached BigData'22 listing)
# -----------------------------------------------------------------------------------------------
# This synthetic generator constructs a *logic-defined* label (INDC) from multivariate time-series:
#   - Each raw variable is a superposition of random-frequency sinusoids with random shifts/scales.
#   - "Relevant" variables R are thresholded as >a, <b, or between (a,b) to produce booleans.
#   - Those booleans are optionally *shifted in time* (disp), then *ANDed within groups (gp)*,
#     and finally *ORed across groups* → an OR-of-ANDs (DNF) gate yields the positive mask.
#   - Sliding, longitudinal windows of length H+1 become rows, labelled by that gate.
#
# That OR-of-ANDs gate is a compact way to *encode interpretable, rule-like structure* in a
# synthetic target—a spirit similar to the "logic engine" that parses/composes rule clauses over
# longitudinal features (from our paper, “LASSO Logic Engine … longitudinal feature learning”
# listed as BigD366 in the IEEE BigData 2022 program’s S3 session). 
#
# This file then *wraps* the generator with two practical calibrations:
#   (1) pick n_cycles so the dataset has *exactly* num_examples rows (no oversampling);
#   (2) tune (thresh_a, thresh_b) so the TRAIN split’s positive prevalence ≈ tr_frac.
#
# Usage:
#   dset, train, test = generate_synthetic_dataset_nexamples(
#       num_examples=3500, tr_frac=0.50,                # targets
#       # everything else is passed straight through to the original generator
#       N=100, S=40, R=(5,6,9,10,15,25,30), AB=("a","a","b","b","a","b","n"),
#       gp=(1,1,1,1,1,1,1), H=10, rseed=1234, train_fr=0.7,
#   )
#

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view  
import pandas as pd
from typing import Sequence, Tuple, Dict, List, Union, Optional

def _cycles_for_examples_guess(num_examples: int, H: int, S: int, n_informative: int) -> int:
    """Rule-of-thumb initial guess for how many cycles are needed to obtain
    approximately `num_examples` rows once the H-history window is formed.

    n_cycles ≈ num_examples / ( H * (S//n_informative + 3) )

    We clamp to at least 3 cycles.
    """
    # Heuristic: more history (H) and more channels (S) increase rows per cycle, so fewer cycles needed.
    # 'n_informative' approximates how many of S have bearing on label construction (len(R)).
    denom = max(1, H * (S // max(1, n_informative) + 3))
    guess = int(np.ceil(num_examples / denom))
    return max(40, guess)  # Safety floor to ensure sufficient raw length for windowing.


def _fit_cycles_to_examples(*, target_examples: int, H: int, S: int, n_informative: int,
                            build_once, rseed: int, thresh_a: float, thresh_b: float,
                            max_iters: int = 10) -> tuple[int, dict]:
    """Find the number of cycles that makes the dataset length equal to
    `target_examples`, using a small number of iterations and *no* oversampling.

    Parameters
    ----------
    build_once : callable
        build_once(n_cycles, thresh_a, thresh_b, rseed)
            -> (dset, dset_train, dset_test)
        Thin wrapper over your original generator.
    Returns
    -------
    n_cycles_final, info_dict
    """
    # Start from the heuristic guess above, then refine multiplicatively (secant-like in log-space).
    n_cyc = _cycles_for_examples_guess(target_examples, H, S, n_informative)
    best = None  # (abs_err, n_examples, n_cyc, seed_shift)
    seed0 = int(rseed)

    for it in range(max_iters):
        # Slight seed shift decorrelates small stochastic effects in lengths due to clipping/random steps.
        dset, dset_train, dset_test = build_once(n_cyc, thresh_a, thresh_b, seed0 + it * 11)
        n_obs = len(dset)
        err = n_obs - target_examples
        cand = (abs(err), n_obs, n_cyc, seed0 + it * 11)
        best = cand if (best is None or cand < best) else best

        if err == 0:
            return n_cyc, {"iters": it+1, "exact": True, "n_examples": n_obs, "seed": seed0 + it*11}

        # Update n_cycles. If current build produced too many rows, step down; else step up.
        # The multiplicative update approximates a one-step secant; nudges avoid getting stuck.
        if n_obs <= 0:
            n_cyc = max(1, n_cyc + 1)
        else:
            new_cyc = int(round(n_cyc * target_examples / n_obs))
            if new_cyc == n_cyc:
                n_cyc += (1 if err < 0 else -1)
                n_cyc = max(1, n_cyc)
            else:
                n_cyc = max(1, new_cyc)

    # If exact match not reached in max_iters, return the best seen; caller may trim rows (never oversample).
    return best[2], {"iters": max_iters, "exact": (best[0]==0), "n_examples": best[1], "seed": best[3]}


def _calibrate_thresholds_to_trfrac(*, tr_frac: float, pos_tol: float,
                                   n_cycles: int, build_once, rseed: int,
                                   thresh_a_init: float, thresh_b_init: float,
                                   max_iters: int = 24,
                                   a_bounds=(1e-3, 0.49), b_bounds=(0.51, 0.999)) -> tuple[float, float, dict]:
    """Tune (thresh_a, thresh_b) so that the *training* split has approximately
    the requested positive fraction `tr_frac`.  This uses a coordinate bisection:
      1) With b fixed, bisection on a.
      2) With a fixed, bisection on b.
    The generator randomness is held fixed by using the same `rseed`.
    """
    # Intuition: Lower 'a' (the "above-threshold" cutoff) makes positives easier to achieve; higher 'b' (the
    # "below-threshold" cutoff) also widens the "positive" region depending on AB mode. We search within safe bounds.
    a_lo, a_hi = map(float, a_bounds)
    b_lo, b_hi = map(float, b_bounds)
    a = float(np.clip(thresh_a_init, a_lo, a_hi))
    b = float(np.clip(thresh_b_init, b_lo, b_hi))

    hist = []  # Keep trace of (a,b,pos_frac) to aid debugging/inspection.

    def _pos_train_fraction(a_val, b_val):
        # Deterministic w.r.t. (n_cycles, rseed) so prevalence responds only to thresholds here.
        dset, dset_train, dset_test = build_once(n_cycles, a_val, b_val, rseed)
        return float(dset_train['INDC'].mean()), len(dset_train), len(dset)

    for it in range(max_iters):
        # --- adjust 'a' with 'b' fixed ---
        # moving 'a' DOWN -> more positives; moving 'a' UP -> fewer positives
        for _ in range(18):
            pos_frac, n_tr, n_total = _pos_train_fraction(a, b)
            hist.append((a, b, pos_frac))
            err = pos_frac - tr_frac
            if abs(err) <= pos_tol:
                return a, b, {"iters": it+1, "trace": hist}
            if err < 0:  # need more positives -> lower a (toward a_lo)
                a_hi = a
                a = 0.5 * (a_lo + a)
            else:        # need fewer positives -> raise a (toward a_hi)
                a_lo = a
                a = 0.5 * (a + a_hi)
            if (a_hi - a_lo) < 1e-4:
                break

        # --- adjust 'b' with 'a' fixed ---
        # moving 'b' UP -> more positives; moving 'b' DOWN -> fewer positives
        for _ in range(18):
            pos_frac, n_tr, n_total = _pos_train_fraction(a, b)
            hist.append((a, b, pos_frac))
            err = pos_frac - tr_frac
            if abs(err) <= pos_tol:
                return a, b, {"iters": it+1, "trace": hist}
            if err < 0:  # need more positives -> raise b (toward b_hi)
                b_lo = b
                b = 0.5 * (b + b_hi)
            else:        # need fewer positives -> lower b (toward b_lo)
                b_hi = b
                b = 0.5 * (b_lo + b)
            if (b_hi - b_lo) < 1e-4:
                break

    # Return best-so-far if tolerance not met. Caller rebuilds with these a,b.
    return a, b, {"iters": max_iters, "trace": hist}


def generate_synthetic_dataset_nexamples(
    *,
    num_examples: int,
    # --- All the same knobs your existing generator accepts ---
    N: int = 100,
    S: int = 40,
    R = (5, 6, 9, 10, 15, 25, 30),
    AB= ("a","a","b","b","a","b","n"),
    gp= (1, 1, 1, 1, 1, 1, 1),
    H: int = 10,
    rseed: int = 1234,
    train_fr: float = 0.70,
    disp: Sequence[int] = (            # time displacements (length S)
        0, 0, 0, 0, 10, 6, 0, 10, 1, 0, 0, 0, 2, 0, 5, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 3, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ),
    # Thresholds are starting points; they will be tuned if `tr_frac` is passed.
    thresh_a: float = 0.20,
    thresh_b: float = 0.75,
    # --- New control targets ---
    tr_frac: float | None = None,   # desired fraction of positives in TRAIN split
    pos_tol: float = 5e-3,          # tolerance on train fraction matching
    # --- Search limits ---
    max_cycle_iters: int = 10,
    max_thresh_iters: int = 24,
    # Pass-through flags
    verbose: bool = True,
):
    """Generate a synthetic dataset of exactly `num_examples` rows without oversampling,
    and (optionally) tune (thresh_a, thresh_b) so that the TRAIN split has
    a desired positive fraction (`tr_frac`).

    This is a *thin wrapper* around `generate_synthetic_dataset(...)`.
    It chooses the number of cycles using a rule-of-thumb and then runs a few iterations
    of multiplicative refinement.  Next, if `tr_frac` is given (default), it applies a
    coordinate bisection to tune (thresh_a, thresh_b).  The random seed is held fixed
    during threshold tuning so the train/test split is deterministic.
    """
    # Validate user target sizes w.r.t. window length
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if num_examples <= H:
        raise ValueError(f"num_examples must exceed H={H} to allow a history window.")

    n_informative = max(1, len(R))

    # Small closure to call the *existing* generator exactly once
    def _build_once(n_cycles, a, b, seed):
        # IMPORTANT: we call general function here.
        return generate_synthetic_dataset(
            N=N, S=S, R=R, AB=AB, gp=gp,
            n_cycles=int(n_cycles), H=H, rseed=int(seed), train_fr=float(train_fr),
            disp=disp, thresh_a=float(a), thresh_b=float(b)
        )

    # --- 1) Pick the number of cycles so len(dset) == num_examples ---
    n_cycles, cyc_info = _fit_cycles_to_examples(
        target_examples=int(num_examples), H=H, S=S, n_informative=n_informative,
        build_once=_build_once, rseed=int(rseed),
        thresh_a=float(thresh_a), thresh_b=float(thresh_b),
        max_iters=int(max_cycle_iters)
    )
    if verbose:
        print(f"[genN] n_cycles → {n_cycles}  (info: {cyc_info})")

    # Build once with the chosen cycles to get a baseline
    dset, dset_train, dset_test = _build_once(n_cycles, thresh_a, thresh_b, rseed)
    n_obs = len(dset)

    # If we're still off, try one ±1 cycle nudge before trimming. This preserves stochasticity profile.
    if n_obs != num_examples:
        # If we're still off (should be rare with the refinement), try one
        # step of ±1 cycles in the right direction before trimming
        if n_obs < num_examples:
            d2, t2, s2 = _build_once(n_cycles + 1, thresh_a, thresh_b, rseed + 17)
            if len(d2) <= num_examples:
                dset, dset_train, dset_test = d2, t2, s2
        elif n_obs > num_examples and n_cycles > 1:
            d2, t2, s2 = _build_once(n_cycles - 1, thresh_a, thresh_b, rseed + 17)
            if len(d2) >= num_examples:
                dset, dset_train, dset_test = d2, t2, s2

    # If still off, *truncate* the overage from the end (never oversample).
    if len(dset) > num_examples:
        over = len(dset) - num_examples
        if verbose:
            print(f"[genN] trimming last {over} rows to hit target={num_examples}")
        dset = dset.iloc[:-over, :].reset_index(drop=True)
        # Rebuild train/test deterministically from the trimmed dset
        # so the requested train_fr is preserved as closely as possible.
        n_tr = int(round(train_fr * len(dset)))
        rng = np.random.default_rng(rseed)
        idx = rng.choice(len(dset), size=n_tr, replace=False)
        mask = np.zeros(len(dset), dtype=bool); mask[idx] = True
        dset_train, dset_test = dset[mask].copy(), dset[~mask].copy()

    # --- 2) If requested, tune thresholds to hit the desired training prevalence ---
    if tr_frac is not None:
        a_star, b_star, tinfo = _calibrate_thresholds_to_trfrac(
            tr_frac=float(tr_frac), pos_tol=float(pos_tol),
            n_cycles=n_cycles, build_once=_build_once, rseed=int(rseed),
            thresh_a_init=float(thresh_a), thresh_b_init=float(thresh_b),
            max_iters=int(max_thresh_iters)
        )
        if verbose:
            print(f"[genN] thresholds → a={a_star:.4f}, b={b_star:.4f}  (info: {tinfo.get('iters')} iters)")
        dset, dset_train, dset_test = _build_once(n_cycles, a_star, b_star, rseed)

    return dset, dset_train, dset_test


def generate_synthetic_dataset(
    N: int = 100,                       # number of base random variables
    S: int = 40,                        # number of superposed variables
    R: Sequence[int] = (5, 6, 9, 10, 15, 25, 30),  # 1-based indices of relevant superposed vars
    AB: Sequence[str] = ("a", "a", "b", "b", "a", "b", "n"),  # threshold modes for each relevant var
    gp: Sequence[int] = (1, 1, 1, 1, 1, 1, 1),    # or-clause grouping (1-based) - default is all AND
    n_cycles: int = 40,                # how many 0-to-pi sinusoid cycles per base signal
    H: int = 10,                       # historical timesteps to capture
    rseed: int = 1234,                 # RNG seed
    train_fr: float = 0.7,             # train/test split fraction
    disp: Sequence[int] = (            # time displacements (length S)
        0, 0, 0, 0, 10, 6, 0, 10, 1, 0, 0, 0, 2, 0, 5, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 3, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ),
    thresh_a: float = 0.20,            # above threshold (fraction of range)
    thresh_b: float = 0.75,            # below threshold (fraction of range)
    return_extra: bool = False         # optionally return debug structures
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
           Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, any]]]:
    """
    Replicates the R scripts synthetic-data generator.

    Returns
    -------
    dset   : pd.DataFrame  full labelled set (chronologically sorted)
    train  : pd.DataFrame  training subset
    test   : pd.DataFrame  testing  subset (non-overlapping with train)

    If `return_extra=True`, the fourth return value is a dict containing
    intermediate objects (`SIG_SIN_RND`, `SIG_SUPPOS`, `time_steps`, etc.)
    that mirror what the R script stored for inspection.
    """
    # ------------------------------------------------------------------------------------------
    # 1) RNG setup and basic helpers
    # ------------------------------------------------------------------------------------------
    np.random.seed(rseed)

    # Helper as in R (keep the same convention): degrees → radians.
    deg2rad = np.deg2rad

    # ------------------------------------------------------------------------------------------
    # 2) Build N random-frequency sinusoid vectors (SIG_SIN_RND)
    # ------------------------------------------------------------------------------------------
    # Each base vector is formed by concatenating `n_cycles` segments of sinusoids over 90..450 degrees,
    # sampled at a random integer step in [1,5].  Then we randomly clip 0..71 samples off the front
    # (phase jitter) and record the minimum length across all N vectors so we can equalize later.
    sig_sin_rnd: List[np.ndarray] = []
    nmin = np.inf                                         # shortest length seen

    for _ in range(N):
        cycles: List[np.ndarray] = []
        for __ in range(n_cycles):
            step = np.random.randint(1, 6)                # 1&5 (inclusive)
            seq_deg = np.arange(90, 450, step, dtype=float)
            cycles.append(np.sin(deg2rad(seq_deg)))
        vec = np.concatenate(cycles)

        # random phase clipping: drop 071 samples from the front
        clip_start = np.random.randint(0, 72)
        vec = vec[clip_start:]

        # keep track of the shortest vector (needed to align lengths for superposition)
        if vec.size < nmin:
            nmin = vec.size
        sig_sin_rnd.append(vec)

    # Equalize lengths and apply a small vertical shift in [0.01, 2.00] to each base vector.
    # Vertical shifts prevent degenerate cancellations when summing.
    for i, vec in enumerate(sig_sin_rnd):
        shift = np.random.randint(1, 201) / 100.0         # 0.01 & 2.00
        sig_sin_rnd[i] = vec[:nmin] + shift

    # ------------------------------------------------------------------------------------------
    # 3) Superpose into S aggregate variables (SIG_SUPPOS)
    # ------------------------------------------------------------------------------------------
    # For each of S variables, pick a random base curve and add `nfun` more random base curves,
    # then apply a random scale.  This yields richer, partially correlated channels.
    nfun = N // S + 1                                      # how many extra curves to add
    sig_sup: List[np.ndarray] = []

    for _ in range(S):
        base = sig_sin_rnd[np.random.randint(0, N)].copy()
        for __ in range(nfun):
            base += sig_sin_rnd[np.random.randint(0, N)]
        scale = (np.random.normal(loc=0.0, scale=0.05) + 0.02) * 100.0
        sig_sup.append(base * scale)

    sig_sup = np.array(sig_sup)      # shape (S, nmin)

    # ------------------------------------------------------------------------------------------
    # 4) Apply per-variable time displacements 
    # ------------------------------------------------------------------------------------------
    # The R script time-shifts certain signals. In this Python version, `SS` is a copy of `sig_sup`.
    # Below, the *labeling condition* itself is shifted forward (via `shift_forward_bool`) rather than
    # shifting the signal values used as features. This design avoids leakage from trivial copies of
    # the target into features when building windows, but is a conscious divergence from the R snippet
    # shown in the docstring block (kept as an inert string here).
    SS = sig_sup.copy()
    
    """
    disp = np.asarray(disp, dtype=int)
    for idx_one_based in R:  # R indices are 1-based
        idx = idx_one_based - 1
        d = disp[idx]
        if d > 0:
            shifted = np.empty_like(SS[idx])
            shifted[:d] = 0
            shifted[d:] = SS[idx, :-d]
            SS[idx] = shifted
    """
    
    # helper to shift a boolean vector forward by d steps
    def shift_forward_bool(mask: np.ndarray, d: int) -> np.ndarray:
        if d <= 0:
            return mask.astype(bool).copy()
        out = np.zeros_like(mask, dtype=bool)
        out[d:] = mask[:-d]
        return out
  
    # ------------------------------------------------------------------------------------------
    # 5) Build OR-of-ANDs mask of positive outcomes (logic gate)
    # ------------------------------------------------------------------------------------------
    # For each relevant series r∈R, compute a boolean condition by thresholding either above 'a',
    # below 'b', or between (a,b) depending on AB[i].  Then shift that *condition* by disp[r].
    # Within each group g (gp[i]), we AND the conditions (all must hold), and across groups we OR.
    # This builds an interpretable DNF-like target, which aligns with logic-parsing approaches to
    # longitudinal features highlighted in our BigData'22 paper.
    total_len = SS.shape[1]
    posg: Dict[int, np.ndarray] = {}
    limits: Dict[str, Tuple[float, float]] = {}
    
    for i, r_one in enumerate(R):
        idx = r_one - 1
        g = gp[i]
        if g not in posg:
            posg[g] = np.ones(total_len, dtype=bool)
    
        vec = sig_sup[idx]                      # <— unshifted baseline series (see note above)
        lo, hi = vec.min(), vec.max()
        #print(f" {idx}")
        d = int(disp[idx])
        name = f"V{r_one}TM{d}"
    
        if AB[i] == "a":
            thr = thresh_a * (hi - lo) + lo
            cond = vec > thr
            limits[name] = (thr, hi)
        elif AB[i] == "b":
            thr = thresh_b * (hi - lo) + lo
            cond = vec < thr
            limits[name] = (lo, thr)
        elif AB[i] == "n":
            thr_lo = thresh_a * (hi - lo) + lo
            thr_hi = thresh_b * (hi - lo) + lo
            cond = (vec > thr_lo) & (vec < thr_hi)
            limits[name] = (thr_lo, thr_hi)
        else:
            raise ValueError("AB entries must be 'a', 'b', or 'n'.")
    
        # Apply time displacement to the condition (not the raw series).
        posg[g] &= shift_forward_bool(cond, d)
    
    # OR across groups to get the final positive mask; also build a simple attribution bitmask.
    positive_mask = np.zeros(total_len, dtype=bool)
    attrib = np.zeros(total_len, dtype=int)
    for g, mask in posg.items():
        positive_mask |= mask
        attrib += mask.astype(int) * (2 ** (g - 1))

    time_steps = np.nonzero(positive_mask)[0]              # 0-based indices where the logic gate fires

    # ------------------------------------------------------------------------------------------
    # 6) Assemble labelled examples (historical windows of length H+1)
    # ------------------------------------------------------------------------------------------
    # For each time t, we pull a *contiguous* block [t-H, …, t] from the S×T matrix (SS),
    # flatten it in variable-major order so columns become V_iTM{lag}, and attach label INDC.
    # Naming convention: TMH, TM(H-1), …, TM0 (TM = "time minus").
    def make_examples(indices: np.ndarray, label: bool) -> pd.DataFrame:
        records = []
        for t in indices:
            records.append(SS[:, t - H : t + 1].flatten(order="C"))  # (S*(H+1),)
        df = pd.DataFrame(
            np.vstack(records),
            columns=[f"V{vi+1}TM{H-j}"      # TMH, TM(H-1), …, TM0
                     for vi in range(S)
                     for j in range(0, H+1)]
        )
        df["INDC"] = label
        df["X"] = indices + 1
        return df

    # Positive rows are those times where the gate fired and we have a full history (t > H).
    pos_idx = time_steps[time_steps > H]
    # Negatives are the complement (also requiring a full history).
    neg_idx = np.setdiff1d(np.arange(total_len), time_steps)
    neg_idx = neg_idx[neg_idx > H]

    pos_df = make_examples(pos_idx, True)
    neg_df = make_examples(neg_idx, False)

    # Concatenate and sort chronologically by X.
    dset = pd.concat([pos_df, neg_df], ignore_index=True).sort_values("X").reset_index(drop=True)

    # ------------------------------------------------------------------------------------------
    # 7) Train/test split
    # ------------------------------------------------------------------------------------------
    # The split is random without stratification here; prevalence control (if desired) is handled
    # upstream by _calibrate_thresholds_to_trfrac in the wrapper.
    n_train = int(round(train_fr * len(dset)))
    train_sel = np.random.choice(dset.index, size=n_train, replace=False)
    dset_train = dset.loc[train_sel].reset_index(drop=True)
    dset_test  = dset.drop(train_sel).reset_index(drop=True)

    if return_extra:
        # Optional debug payload to inspect the generative internals.
        extra = dict(
            SIG_SIN_RND=sig_sin_rnd,
            SIG_SUPPOS=sig_sup,
            SS=SS,
            time_steps=time_steps,
            limits=limits,
            attrib=attrib,
        )
        return dset, dset_train, dset_test, extra
    else:
        return dset, dset_train, dset_test

# ----------------------------------------------------------------------------------------------
# Example script entry point
# ----------------------------------------------------------------------------------------------
# Below is a simple end-to-end invocation that:
#   • Generates approximately NUM_EXAMPLES rows (exactly, after trimming if needed),
#   • Tunes thresholds to target an X% positive fraction *in the TRAIN split*,
#   • Prints a preview and saves CSVs.
if __name__ == "__main__":
    NUM_EXAMPLES = 8000
  
    full, train, test = generate_synthetic_dataset_nexamples(
        num_examples = NUM_EXAMPLES,
        # --- Various knobs that will affect the generator ---
        N = 100,
        S = 40,
        R = (5, 6, 9, 10, 15, 25, 30),
        AB= ("a","a","b","b","a","b","n"),
        gp= (1, 1, 1, 1, 1, 1, 1),
        H = 10,
        rseed = 1234,
        train_fr = 0.70,
        disp = (            # time displacements (length S)
            0, 0, 0, 0, 10, 6, 0, 10, 1, 0, 0, 0, 2, 0, 5, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 3, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        # Thresholds are starting points; they will be tuned if `tr_frac` is passed.
        thresh_a = 0.20,
        thresh_b = 0.75,
        # --- Control targets ---
        tr_frac = 0.25,          # desired fraction of positives in TRAIN split
        pos_tol = 5e-3,          # tolerance on train fraction matching
        # --- Search limits ---
        max_cycle_iters = 10,
        max_thresh_iters = 24,
        # Pass-through flag
        verbose = True)

    print(train.head())
    print(test.head())

    # save to CSV 
    train.to_csv("train_data.csv", index=False)
    test.to_csv("test_data.csv",  index=False)
