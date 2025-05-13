from collections import defaultdict
from sklearn.neighbors import KernelDensity
import numpy as np

class KDENaiveBayes:
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde_models = defaultdict(dict)
        self.class_priors = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / len(y)
            for feature_index in range(X.shape[1]):
                feature_vals = X_cls[:, feature_index]
                zero_mask = feature_vals == 0
                zero_fraction = np.mean(zero_mask)
                if zero_fraction > 0.10:
                    feature_vals = feature_vals[~zero_mask]
                kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
                kde.fit(feature_vals.reshape(-1, 1))
                self.kde_models[cls][feature_index] = kde

    def predict_log_proba(self, X):
        log_probs = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            log_prior = np.log(self.class_priors[cls])
            total_log_likelihood = np.zeros(X.shape[0])
            for feature_index in range(X.shape[1]):
                feature_vals = X[:, feature_index].reshape(-1, 1)
                valid_mask = ~np.isnan(feature_vals).flatten()
                kde = self.kde_models[cls][feature_index]

                log_dens = np.full(X.shape[0], 0.0)  # neutral log-probability
                if valid_mask.any():
                    log_dens[valid_mask] = kde.score_samples(feature_vals[valid_mask])
                
                total_log_likelihood += log_dens
            log_probs[:, i] = log_prior + total_log_likelihood
        return log_probs

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]
