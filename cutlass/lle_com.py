# Custom L1-regularized logistic regression tailored for logic-based feature selection
# Implements the method described in the paper "Efficient Longitudinal Feature Selection with Binarized Data Transformation"
# This approach enhances support recovery by using binarized features and leveraging the irrepresentability condition (IC)

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse, csr_matrix
# for comparison
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score
# for defining the model class
from sklearn.base import BaseEstimator, ClassifierMixin
# for timing
import timeit

class LLEClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn-compatible classifier using L1-regularized logistic regression on binarized feature vectors.

    Parameters:
    - lambda_ (float): Regularization strength for the L1 penalty.
    - learning_rate (float): Step size for gradient descent.
    - max_iter (int): Maximum number of iterations for training.
    - k_folds (int): Number of cross-validation folds.
    - batch_size (int): Number of samples per gradient descent mini-batch.
    - verbose (bool): Print training diagnostics if True.
    - early_exit (bool or float): Exit training early if performance threshold is met.
    """
    def __init__(self, lambda_=0.01, learning_rate=0.1, max_iter=100, k_folds=3, batch_size=256, verbose=False, early_exit=False):
        # Store user-defined parameters
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.verbose = verbose
        self.early_exit = early_exit
        # Placeholder for learned coefficients (includes intercept)
        self.coef_ = None

    def fit(self, X, y):
        """
        Fit the LLEClassifier model using L1-regularized logistic regression.

        Args:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Binary target vector.

        Returns:
        - self: Fitted estimator.
        """
        # Delegate actual training logic to lle() function
        self.coef_ = lle(
            X=X,
            y=y,
            lambda_=self.lambda_,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            k_folds=self.k_folds,
            batch_size=self.batch_size,
            verbose=self.verbose,
            early_exit=self.early_exit
        )
        return self

    def predict(self, X):
        """
        Predict binary class labels for input data.

        Args:
        - X (np.ndarray): Feature matrix.

        Returns:
        - np.ndarray: Binary prediction vector.
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
        # Add intercept term to feature matrix
        X_aug = np.hstack((np.ones((X.shape[0], 1)), X))
        # Predict using sigmoid of dot product
        probs = sigmoid(X_aug.dot(self.coef_))
        return (probs >= 0.5).astype(int)


def sigmoid(s):
    """
    Compute the logistic sigmoid function.

    Args:
    - s (np.ndarray): Input vector.

    Returns:
    - np.ndarray: Sigmoid of input.
    """
    return 1 / (1 + np.exp(-s))


def prox_L1(w, alpha):
    """
    Apply the proximal L1 operator (soft thresholding).

    Args:
    - w (np.ndarray): Weight vector.
    - alpha (float): Threshold for shrinkage.

    Returns:
    - np.ndarray: Regularized weight vector.
    """
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def compute_partial_gradient(X_batch, y_batch, w):
    """
    Compute the gradient of the logistic loss for a mini-batch.

    Args:
    - X_batch (np.ndarray): Mini-batch of feature data.
    - y_batch (np.ndarray): Mini-batch of target values.
    - w (np.ndarray): Weight vector.

    Returns:
    - np.ndarray: Gradient vector.
    """
    s = X_batch.dot(w)                 # Linear transformation
    p = sigmoid(s)                     # Probabilities via sigmoid
    grad = X_batch.T.dot(p - y_batch)  # Derivative of loss function
    return grad


def lasso_intercept(X, y, w):
    """
    Compute intercept for logistic model to center prediction around target mean.

    Args:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - w (np.ndarray): Weight vector (excluding intercept).

    Returns:
    - float: Calculated intercept.
    """
    # Average of each feature
    feature_means = np.mean(X, axis=0)
    # Calculate the empirical probability of class 1 (response vector)
    p = np.mean(y)
    # Calculate the linear combination of coefficients and feature means
    linear_combo = np.dot(w, feature_means)
    # Calculate the intercept
    return(np.log(p / (1 - p)) - linear_combo)


def calculate_youden_j(y_true, y_pred, thresh=0.5):
    """
    Compute Youden's J-index: sensitivity + specificity - 1. Youden's index is superior
    to many other measures (including accuracy and F1) for imbalanced data sets and is at
    the same time equally as good those others with balanced data sets. As a result, the
    J-index is preferred as a comparison metric.

    Args:
    - y_true (np.ndarray): Ground truth binary labels.
    - y_pred (np.ndarray): Predicted binary labels.
    - thresh (float): Threshold to convert probabilities to binary class (default 0.5).

    Returns:
    - float: Youden's J statistic.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # True Positives (TP): Correctly predicted positives
    tp = np.sum((y_true >= thresh) & (y_pred >= thresh))
    # True Negatives (TN): Correctly predicted negatives
    tn = np.sum((y_true < thresh) & (y_pred < thresh))
    # False Positives (FP): Incorrectly predicted positives
    fp = np.sum((y_true < thresh) & (y_pred >= thresh))
    # False Negatives (FN): Incorrectly predicted negatives
    fn = np.sum((y_true >= thresh) & (y_pred < thresh))

    # Sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Return Youden's J-Index
    return sensitivity + specificity - 1


def lle(X, y, lambda_, learning_rate, max_iter, k_folds=3, batch_size=256, verbose=False, early_exit=False):
    """
    Perform logistic regression with L1 regularization on binarized features using k-fold cross-validation.

    Args:
    - X (np.ndarray or sparse): Feature matrix.
    - y (np.ndarray): Binary response vector.
    - lambda_ (float): L1 regularization strength.
    - learning_rate (float): Learning rate for updates.
    - max_iter (int): Max gradient iterations per fold.
    - k_folds (int): Number of CV folds.
    - batch_size (int): Mini-batch size for updates.
    - verbose (bool): Print training status if True.
    - early_exit (bool or float): Stop training early if threshold J is reached.

    Returns:
    - np.ndarray: Final coefficient vector (including intercept).
    """
    N = X.shape[0]

    # Step 1: Add intercept column to feature matrix.
    # This ensures the learned model can shift the decision boundary independently of input features.
    if issparse(X):
        X_augmented = csr_matrix(np.hstack((np.ones((N, 1)), X.toarray())))
    else:
        X_augmented = np.hstack((np.ones((N, 1)), X))

    n_features = X_augmented.shape[1]
    # Initialize coefficients including intercept to zero
    w = np.zeros(n_features)

    # Step 2: Create k folds for cross-validation by shuffling and splitting indices
    indices = np.arange(N)
    np.random.shuffle(indices)
    fold_sizes = (N // k_folds) * np.ones(k_folds, dtype=int)
    # Distribute the remainder among the first folds
    fold_sizes[:N % k_folds] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        folds.append(indices[start:stop])
        current = stop

    best_j_index = -np.inf
    best_w = None

    # Step 3: Train a model for each fold using stochastic gradient descent
    for fold_idx, val_indices in enumerate(folds):
        train_indices = np.setdiff1d(indices, val_indices)
        X_train_fold, X_val_fold = X_augmented[train_indices], X_augmented[val_indices]
        y_train_fold, y_val_fold = y[train_indices], y[val_indices]

        # Initialize weights for each fold
        w_fold = np.zeros(n_features)
        num_batches = int(np.ceil(len(train_indices) / batch_size))

        for iteration in range(1, max_iter + 1):
            # Shuffle the training set each epoch to improve convergence stability
            train_perm = np.random.permutation(len(train_indices))
            X_shuffled, y_shuffled = X_train_fold[train_perm], y_train_fold[train_perm]

            for batch_idx in range(num_batches):
                # Slice out current mini-batch
                start, end = batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(train_indices))
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]

                # Compute partial gradient of the logistic loss
                grad = compute_partial_gradient(X_batch, y_batch, w_fold) / (end - start)
                # Take a gradient descent step, then apply L1 shrinkage
                w_temp = w_fold - learning_rate * grad
                w_fold = prox_L1(w_temp, learning_rate * lambda_)

            # Step 4: Periodically evaluate on validation set using Youden's J-index
            if early_exit and (iteration % 10 == 0):
                y_pred_prob = sigmoid(X_val_fold.dot(w_fold))
                y_pred = (y_pred_prob >= 0.5).astype(int)
                youden_j = calculate_youden_j(y_val_fold, y_pred)

                if verbose:
                    print(f"Fold {fold_idx+1}, Iter {iteration}, Youden's J: {youden_j:.3f}")

                # Track best-performing fold weights
                if youden_j > best_j_index:
                    best_j_index = youden_j
                    best_w = w_fold.copy()

                # Exit early if target J-index reached
                if youden_j >= early_exit:
                    break

            # Early exit condition
            if youden_j >= early_exit:
                break

        # Accumulate weight vectors across folds
        w += w_fold

        # Early exit condition - allows breaking from the outer loop after a single early exit condition is met rather than
        # having to cycle through all folds first.
        #if youden_j >= early_exit:
        #    if verbose:
        #        print(f"Early exit at fold {fold_idx+1}, iteration {iteration}, youden {youden_j}.")
        #        break

    # Step 5: Average the accumulated weight vectors to produce final coefficients
    w /= k_folds

    # Step 6: Post-processing for logical sparsity: promote interpretability by thresholding
    sw = np.sort(np.abs(w[1:]))[::-1]  # Skip intercept (w[0])
    for threshold in sw:
        nw = w[1:].copy()
        # Set small coefficients to 0, large ones to logical proxy (+/-10)
        nw[np.abs(nw) < threshold] = 0
        nw[np.abs(nw) >= threshold] = 10 * np.sign(nw[np.abs(nw) >= threshold])
        # Recalculate intercept to reflect thresholding
        I = lasso_intercept(X, y, nw)

        # Check if this binarized representation preserves predictive performance
        y_pred = (sigmoid(np.hstack((np.ones((X.shape[0], 1)), X)).dot(np.append(I, nw))) >= 0.5).astype(int)
        youden_j = calculate_youden_j(y, y_pred)

        if verbose:
            print(f"Post-proc threshold: {threshold:.4f}, Youden's J: {youden_j:.3f}")

        if youden_j >= early_exit:
            w = np.append(I, nw)
            break

    # Final weight vector includes postprocessed intercept and selected logical weights
    return w
