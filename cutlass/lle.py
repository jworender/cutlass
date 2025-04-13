# for the custom logistic regression
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
    def __init__(self, lambda_=0.01, learning_rate=0.1, max_iter=100, k_folds=3, batch_size=256, verbose=False, early_exit=False):
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.verbose = verbose
        self.early_exit = early_exit
        self.coef_ = None

    def fit(self, X, y):
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
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
        X_aug = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept
        probs = sigmoid(X_aug.dot(self.coef_))
        return (probs >= 0.5).astype(int)

# compute the sigmoid function
def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def prox_L1(w, alpha):
    """Apply the proximal operator for L1 regularization (soft thresholding)."""
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)

# Compute the gradient of the cost function for a batch of data

# Parameters:
#    - X_batch: Feature matrix for the batch (batch_size x n_features)
#    - y_batch: Response vector for the batch (batch_size,)
#    - w: Current weight vector (n_features,)
# Returns:
#    - grad: Gradient vector for the batch (n_features,)

def compute_partial_gradient(X_batch, y_batch, w):
    s = X_batch.dot(w)  # Compute s^(i) = w^T x^(i)
    p = sigmoid(s)      # Compute σ(s^(i))
    grad = X_batch.T.dot(p - y_batch)  # Sum over batch
    return grad

# Calculate the intercept from known LASSO logistic regression coefficients, this is useful for initiating an early exit and setting the coefficients equal to a limiting value

# Parameters:
#    - X: The feature matrix (n_examples x n_features)
#    - y: The response vector (n_examples)
#    - w: The weight vector (without the intercept included)
# Returns:
#    - w0: The intercept.
def lasso_intercept(X, y, w):
    # calculate the means of all features
    feature_means = np.mean(X, axis=0)
    # calculate the mean probability of the response vector
    p = np.mean(y)
    # calculate the linear combination of coefficients and feature means
    linear_combo = np.dot(w, feature_means)
    # Calculate the intercept
    return(np.log(p / (1 - p)) - linear_combo)

# Calculate Youden's J-Index

# Parameters:
#    - y_true: The response vector (ground truth).
#    - y_pred: The vector of predicted values (calculated).
#    - thresh: The threshold above which a predicted value is assessed as True.
# Returns:
#    - youden_j: Youden's J-Index

def calculate_youden_j(y_true, y_pred, thresh=0.5):
    # Ensure inputs are NumPy arrays
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

    # Youden's J-Index
    youden_j = sensitivity + specificity - 1

    return youden_j

# Perform logistic regression with L1 regularization using gradient descent and cross-validation early stopping

# Parameters:
#    - X: Feature matrix (n_samples x n_features)
#    - y: Response vector (n_samples,)
#    - lambda_: Regularization parameter
#    - learning_rate: Learning rate for gradient descent
#    - max_iter: Maximum number of iterations
#    - k_folds: The number of folds to use during cross-validation
#    - batch_size: The size of the batches to use for calculating partial gradients
#    - verbose: Boolean which determines whether to display informational messages
#    - early_exit: False to prevent early exit, True or a floating point threshold to early exit at 100% or the specified accuracy threshold
# Returns:
#    - w: Estimated weight vector (n_features,)
 
def lle(X, y, lambda_, learning_rate, max_iter, k_folds=3, batch_size=256, verbose=False, early_exit=False):
    N = X.shape[0]
    intercept = True  # Include intercept term
    if intercept:
        # Add intercept term
        if issparse(X):
            X_augmented = csr_matrix(np.hstack((np.ones((N, 1)), X.toarray())))
        else:
            X_augmented = np.hstack((np.ones((N, 1)), X))
    else:
        X_augmented = X

    n_features = X_augmented.shape[1]
    w = np.zeros(n_features)

    # Implement k-fold cross-validation manually
    indices = np.arange(N)
    np.random.shuffle(indices)
    fold_sizes = (N // k_folds) * np.ones(k_folds, dtype=int)
    fold_sizes[:N % k_folds] += 1  # Distribute the remainder among the first folds
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        folds.append(indices[start:stop])
        current = stop

    youden_j = 0.0
    best_w = None
    best_j_index = -np.inf

    for fold_idx, val_indices in enumerate(folds):
        train_indices = np.setdiff1d(indices, val_indices)
        X_train_fold, X_val_fold = X_augmented[train_indices], X_augmented[val_indices]
        y_train_fold, y_val_fold = y[train_indices], y[val_indices]

        # Initialize weights for each fold
        w_fold = np.zeros(n_features)
        num_batches = int(np.ceil(len(train_indices) / batch_size))

        for iteration in range(1, max_iter + 1):
            # Shuffle training data
            train_perm = np.random.permutation(len(train_indices))
            X_shuffled = X_train_fold[train_perm]
            y_shuffled = y_train_fold[train_perm]

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(train_indices))
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Compute gradient
                grad = compute_partial_gradient(X_batch, y_batch, w_fold) / (end - start)

                # Gradient descent update with L1 regularization
                w_temp = w_fold - learning_rate * grad
                w_fold = prox_L1(w_temp, learning_rate * lambda_)

            # Early exit evaluation using validation set
            if early_exit and (iteration % 10 == 0):
                s_val = X_val_fold.dot(w_fold)
                y_pred_prob = sigmoid(s_val)
                y_pred = (y_pred_prob >= 0.5).astype(int)
                youden_j = calculate_youden_j(y_val_fold, y_pred)

                if verbose:
                    print(f"Fold {fold_idx+1}, Iteration {iteration}, Youden's J-Index: {youden_j}")

                # Check if current model is better
                if youden_j > best_j_index:
                    best_j_index = youden_j
                    best_w = w_fold.copy()

                # Early exit condition
                if youden_j >= early_exit:
                     break

            # Early exit condition
            if youden_j >= early_exit:
                break

        # Update weights after each fold
        w += w_fold

        # Early exit condition - allows breaking from the outer loop after a single early exit condition is met rather than
        # having to cycle through all folds first.
        #if youden_j >= early_exit:
        #    if verbose:
        #        print(f"Early exit at fold {fold_idx+1}, iteration {iteration}, youden {youden_j}.")
        #        break

    # Average weights over folds
    w /= k_folds

    #print(w)
    # Post-processing: Enforce binary weights
    sw = np.sort(np.abs(w[1:]))[::-1]  # Exclude intercept
    for threshold in sw:
        nw = w[1:].copy()
        nw[np.abs(w[1:]) < threshold] = 0
        nw[np.abs(w[1:]) >= threshold] = 10 * np.sign(w[1:][np.abs(w[1:]) >= threshold])  # Preserve sign
        I = lasso_intercept(X, y, nw)
        y_pred = (sigmoid(np.hstack((np.ones((X.shape[0], 1)), X)).dot(np.append(I, nw))) >= 0.5).astype(int)
        youden_j = calculate_youden_j(y, y_pred)
        if verbose:
            print(f"Post-processing threshold: {threshold}, Youden's J-Index: {youden_j} ({early_exit})")
        if youden_j >= early_exit:
            w = np.append(I, nw)
            break

    return w


