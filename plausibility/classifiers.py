import numpy as np
from sklearn.base import BaseEstimator
import sklearn.metrics as metrics
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from plausibility.metrics import error_beta_score

class OCSVM(BaseEstimator):
    def __init__(self, nu=.5, gamma='scale', verbose=False):
        self.nu = nu
        self.gamma = gamma
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the One-Class SVM model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values. Expected to be binary with 1 indicating the positive class.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.oc_classifier_ = OneClassSVM(nu=self.nu,
                                          gamma=self.gamma,
                                          verbose=self.verbose,
                                          cache_size=7000)
        X_pos = np.array([x for x, l in zip(X, y) if l == 1])
        self.oc_classifier_.fit(X_pos)
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels for each input sample.

        Raises
        ------
        NotFittedError
            If the classifier is not fitted yet.
        """
        check_is_fitted(self, 'oc_classifier_')
        X = check_array(X)
        return self.oc_classifier_.predict(X)

    def score(self, X, y, beta):
        """
        Compute the error beta score for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            The error beta score for the provided data.

        Raises
        ------
        NotFittedError
            If the classifier is not fitted yet.
        """
        check_is_fitted(self, 'oc_classifier_')
        X, y = check_X_y(X, y)

        y_hat = self.predict(X)
        cm = metrics.confusion_matrix(y, y_hat, labels=[-1, 1])
        tn, fp, fn, tp = cm.ravel()
    
        # accuracy = (tp + tn) / (tp + fp + fn + tn) 
        # precision = tp / (tp + fp) if tp + fp > 0 else np.nan
        # recall = tp / (tp + fn) if tp + fn > 0 else np.nan
        # specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
        false_pos_rate = fp / (tn + fp) if tn + fp > 0 else np.nan
        false_neg_rate = fn / (fn + tp) if fn + tp > 0 else np.nan
        return error_beta_score(false_pos_rate, false_neg_rate, beta)


    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        
        self.oc_classifier_ = OneClassSVM(nu=self.nu, gamma=self.gamma)
        return self
    
    def plausibility(self, X):
        """
        Compute the plausibility scores for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : array-like of shape (n_samples,)
            The plausibility scores for each input sample.

        Raises
        ------
        NotFittedError
            If the classifier is not fitted yet.
        """
        check_is_fitted(self, 'oc_classifier_')
        return self.oc_classifier_.decision_function(X)