# baseline/baseline_model.py
from __future__ import annotations

import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ================= 路径配置 =================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 尝试导入 load_data 以便单独测试
try:
    from data_preprocessing import load_data
except ImportError:
    pass


def run_baseline(X_train, y_train, X_valid, y_valid, X_test, seed=42):
    # 逻辑回归
    lr = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    lr.fit(X_train, y_train)
    lr_valid = lr.predict_proba(X_valid)[:, 1]
    lr_test = lr.predict_proba(X_test)[:, 1]

    # 随机森林
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    rf_valid = rf.predict_proba(X_valid)[:, 1]
    rf_test = rf.predict_proba(X_test)[:, 1]

    w_lr, w_rf = 0.3, 0.7
    valid_pred = w_lr * lr_valid + w_rf * rf_valid
    test_pred = w_lr * lr_test + w_rf * rf_test

    metrics = {"auc": float(roc_auc_score(y_valid, valid_pred))}
    return metrics, valid_pred, test_pred