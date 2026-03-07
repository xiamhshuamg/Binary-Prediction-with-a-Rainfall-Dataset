# gbdt/gbdt_model.py
from __future__ import annotations

import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _import_gbdt_libs():
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    return xgb, lgb, CatBoostClassifier

def run_gbdt(X_train, y_train, X_valid, y_valid, X_test, seed=42):
    xgb, lgb, CatBoostClassifier = _import_gbdt_libs()

    # XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.6,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=200,
        n_jobs=-1,
        random_state=seed # <--- 使用 seed
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    # LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.015,
        num_leaves=32,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.6,
        objective="binary",
        metric="auc",
        extra_trees=True,
        random_state=seed, # <--- 使用 seed
        n_jobs=-1,
        verbose=-1
    )
    # Note: LightGBM python API callbacks update
    try:
        callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
        model_lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=callbacks)
    except:
        model_lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=200, verbose=False)


    # CatBoost
    model_cb = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed, # <--- 使用 seed
        verbose=False,
        allow_writing_files=False
    )
    model_cb.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=200)

    # 预测
    px_v = model_xgb.predict_proba(X_valid)[:, 1]
    pl_v = model_lgb.predict_proba(X_valid)[:, 1]
    pc_v = model_cb.predict_proba(X_valid)[:, 1]

    px_t = model_xgb.predict_proba(X_test)[:, 1]
    pl_t = model_lgb.predict_proba(X_test)[:, 1]
    pc_t = model_cb.predict_proba(X_test)[:, 1]

    valid_pred = (px_v + pl_v + pc_v) / 3
    test_pred = (px_t + pl_t + pc_t) / 3

    metrics = {"auc": float(roc_auc_score(y_valid, valid_pred))}
    return metrics, valid_pred, test_pred