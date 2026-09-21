"""
Classical Machine Learning Baselines for MSTAR SAR ATR.

Modernized refactoring of Project_21C02.ipynb using contemporary scikit-learn.
Supports PCA dimensionality reduction combined with SVM, Random Forest, GBDT, and MLP.
"""

from typing import Dict, Optional, Tuple, Union
import pickle
import numpy as np


class ClassicalSARPipeline:
    """
    Classical ML Pipeline for SAR Image Classification:
        Raw Pixel Array -> Normalization & Mean-centering -> PCA Reduction -> Classifier
    """

    SUPPORTED_CLASSIFIERS = ["svm", "rf", "gbdt", "mlp", "dt", "knn", "bayes"]

    def __init__(
        self,
        classifier_type: str = "svm",
        n_pca_components: int = 80,
        random_state: int = 42,
        **classifier_kwargs,
    ) -> None:
        self.classifier_type = classifier_type.lower()
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        self.classifier_kwargs = classifier_kwargs

        self.pca = None
        self.classifier = self._init_classifier()
        self.is_fitted = False

    def _init_classifier(self):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        if self.classifier_type == "svm":
            return SVC(
                C=self.classifier_kwargs.get("C", 1.0),
                kernel=self.classifier_kwargs.get("kernel", "rbf"),
                probability=True,
                random_state=self.random_state,
            )
        elif self.classifier_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.classifier_kwargs.get("n_estimators", 200),
                max_features=self.classifier_kwargs.get("max_features", "sqrt"),
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.classifier_type == "gbdt":
            return GradientBoostingClassifier(
                n_estimators=self.classifier_kwargs.get("n_estimators", 150),
                learning_rate=self.classifier_kwargs.get("learning_rate", 0.05),
                random_state=self.random_state,
            )
        elif self.classifier_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=self.classifier_kwargs.get("hidden_layer_sizes", (128, 64)),
                activation="relu",
                max_iter=500,
                random_state=self.random_state,
            )
        elif self.classifier_type == "dt":
            return DecisionTreeClassifier(
                criterion="entropy",
                random_state=self.random_state,
            )
        elif self.classifier_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=self.classifier_kwargs.get("n_neighbors", 5),
                weights="distance",
            )
        elif self.classifier_type == "bayes":
            return GaussianNB()
        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        from sklearn.decomposition import PCA

        X_flat = np.asarray(X, dtype=np.float32)
        if X_flat.ndim > 2:
            X_flat = X_flat.reshape(X_flat.shape[0], -1)

        # Scale to [0, 1] if not already
        if X_flat.max() > 1.0:
            X_flat = X_flat / 255.0

        # Mean-wise centering per sample
        mean_per_sample = np.mean(X_flat, axis=1, keepdims=True)
        X_centered = X_flat - mean_per_sample

        # PCA transformation
        if self.n_pca_components is not None and self.n_pca_components > 0:
            max_components = min(X_centered.shape[0], X_centered.shape[1])
            actual_n = min(self.n_pca_components, max_components)
            if fit or self.pca is None:
                self.pca = PCA(n_components=actual_n, random_state=self.random_state)
                X_transformed = self.pca.fit_transform(X_centered)
            else:
                X_transformed = self.pca.transform(X_centered)
            return X_transformed

        return X_centered

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalSARPipeline":
        """Fits the PCA and classical classifier."""
        X_feats = self._preprocess(X, fit=True)
        self.classifier.fit(X_feats, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class indices."""
        X_feats = self._preprocess(X, fit=False)
        return self.classifier.predict(X_feats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities if supported."""
        X_feats = self._preprocess(X, fit=False)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_feats)
        elif hasattr(self.classifier, "decision_function"):
            df = self.classifier.decision_function(X_feats)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        raise NotImplementedError("Classifier does not support probability estimation.")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns accuracy score on test set."""
        X_feats = self._preprocess(X, fit=False)
        return float(self.classifier.score(X_feats, y))

    def save(self, filepath: str) -> None:
        """Saves pipeline state to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "ClassicalSARPipeline":
        """Loads pipeline from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
