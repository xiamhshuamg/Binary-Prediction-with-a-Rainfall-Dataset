# nn_ensemble/nn_ensemble_model.py
from __future__ import annotations

import os
import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_nn_ensemble(X_train, y_train, X_valid, y_valid, X_test, seed=42):

    # KNN
    knn = KNeighborsClassifier(n_neighbors=150, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)
    p_knn_v = knn.predict_proba(X_valid)[:, 1]
    p_knn_t = knn.predict_proba(X_test)[:, 1]

    # ExtraTrees
    et = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=6,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=seed,
        n_jobs=-1
    )
    et.fit(X_train, y_train)
    p_et_v = et.predict_proba(X_valid)[:, 1]
    p_et_t = et.predict_proba(X_test)[:, 1]

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=0.1,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed  # <--- 使用 seed
    )
    mlp.fit(X_train, y_train)
    p_mlp_v = mlp.predict_proba(X_valid)[:, 1]
    p_mlp_t = mlp.predict_proba(X_test)[:, 1]

    valid_pred = (p_knn_v + p_et_v + p_mlp_v) / 3
    test_pred = (p_knn_t + p_et_t + p_mlp_t) / 3

    metrics = {"auc": float(roc_auc_score(y_valid, valid_pred))}
    return metrics, valid_pred, test_pred