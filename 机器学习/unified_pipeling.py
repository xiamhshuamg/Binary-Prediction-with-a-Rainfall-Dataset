# unified_pipeline.py
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_preprocessing import read_raw_data, Preprocessor

# Import models
from baseline.baseline_model import run_baseline
from gbdt.gbdt_model import run_gbdt
from nn_ensemble.nn_ensemble_model import run_nn_ensemble


def get_day_col(df):
    if "day" in df.columns:
        return df["day"].values
    return np.zeros(len(df))


def run_phase(X, y, X_test, groups, n_splits=5, phase_name="Phase 1", seeds=[42]):
    """
    支持多 Seed 平均
    """
    # 容器：累加所有 Seed 的结果
    oof_preds_total = np.zeros((len(y), 3), dtype=float)
    test_preds_total = np.zeros((len(X_test), 3), dtype=float)

    print(f"\n--- {phase_name} Start (Train={len(X)}, Seeds={len(seeds)}) ---")

    for seed in seeds:
        print(f"  > Running Seed: {seed}")
        # 使用 StratifiedGroupKFold 以保证 Fold 分布更均匀，减少方差
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # 单次 Seed 的容器
        oof_preds_seed = np.zeros((len(y), 3), dtype=float)
        test_preds_seed = np.zeros((len(X_test), 3), dtype=float)

        for i, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups), 1):
            X_tr_raw = X.iloc[tr_idx].copy()
            y_tr = y.iloc[tr_idx]
            X_va_raw = X.iloc[va_idx].copy()
            y_va = y.iloc[va_idx]

            # Fit Preprocessor
            pre = Preprocessor(n_components=1, n_clusters=6).fit(X_tr_raw)
            X_tr = pre.transform(X_tr_raw)
            X_va = pre.transform(X_va_raw)
            X_te = pre.transform(X_test)

            # Train with current SEED
            _, pvb, ptb = run_baseline(X_tr, y_tr, X_va, y_va, X_te, seed=seed)
            _, pvg, ptg = run_gbdt(X_tr, y_tr, X_va, y_va, X_te, seed=seed)
            _, pvn, ptn = run_nn_ensemble(X_tr, y_tr, X_va, y_va, X_te, seed=seed)

            oof_preds_seed[va_idx, 0] = pvb
            oof_preds_seed[va_idx, 1] = pvg
            oof_preds_seed[va_idx, 2] = pvn

            test_preds_seed[:, 0] += ptb
            test_preds_seed[:, 1] += ptg
            test_preds_seed[:, 2] += ptn

        # Seed 内部 normalize test preds (除以 n_splits)
        test_preds_seed /= n_splits

        # 累加到 Total
        oof_preds_total += oof_preds_seed
        test_preds_total += test_preds_seed

        # 计算当前 Seed 的 AUC
        avg_seed_p = np.mean(oof_preds_seed, axis=1)
        seed_auc = roc_auc_score(y, avg_seed_p)
        print(f"  > Seed {seed} Overall AUC: {seed_auc:.5f}")

    # 取所有 Seeds 的平均
    oof_preds_final = oof_preds_total / len(seeds)
    test_preds_final = test_preds_total / len(seeds)

    avg_p_final = np.mean(oof_preds_final, axis=1)
    final_auc = roc_auc_score(y, avg_p_final)
    print(f"[{phase_name}] Ensembled ({len(seeds)} seeds) AUC: {final_auc:.5f}")

    return oof_preds_final, test_preds_final, final_auc


def run_unified(
        n_splits: int = 5,
        output_path: str = "submission.csv",
):
    X_raw, y, X_test_raw, test_ids = read_raw_data()
    y = y.values.astype(int)
    groups = get_day_col(X_raw)

    # ==========================
    # Phase 1: 多 Seed 训练
    # ==========================
    # 使用 3 个不同的 Seed 来消除波动
    SEEDS = [42, 2023, 1024]

    oof_p1, test_p1, auc_p1 = run_phase(
        X_raw, pd.Series(y), X_test_raw, groups,
        n_splits, "Phase 1", seeds=SEEDS
    )

    # ==========================
    # 保存 Phase 1 结果（伪标签版本）
    # ==========================
    print("\n[Saving Phase 1 Results (Pseudo-Label Version)]")

    # Phase 1 的 Rank 融合
    FIXED_WEIGHTS = np.array([0.00, 0.51, 0.49])  # [Base, GBDT, NN]

    # 计算 Phase 1 测试集的 Rank
    test_ranks_p1 = np.zeros_like(test_p1)
    for k in range(3):
        test_ranks_p1[:, k] = rankdata(test_p1[:, k]) / len(test_p1)

    # 应用固定权重融合（Phase 1 版本）
    final_pred_rank_p1 = np.dot(test_ranks_p1, FIXED_WEIGHTS)
    final_pred_rank_p1 = np.clip(final_pred_rank_p1, 0.0, 1.0)

    # 保存 Phase 1 的 Rank 融合结果
    sub_pseudo = pd.DataFrame({
        "id": test_ids.values if hasattr(test_ids, "values") else test_ids,
        "rainfall": final_pred_rank_p1
    })
    sub_pseudo.to_csv("submission_pseudo_rank.csv", index=False)
    print("Saved submission_pseudo_rank.csv (Phase 1 Rank Fusion - Pseudo-Label Version)")

    # Phase 1 的概率平均作为对比
    final_pred_prob_p1 = 0.4 * test_p1[:, 1] + 0.6 * test_p1[:, 2]

    sub_pseudo_prob = pd.DataFrame({
        "id": test_ids.values if hasattr(test_ids, "values") else test_ids,
        "rainfall": final_pred_prob_p1
    })
    sub_pseudo_prob.to_csv("submission_pseudo_prob.csv", index=False)
    print("Saved submission_pseudo_prob.csv (Phase 1 Probability Fusion)")

    print(f"[Phase 1 Results Saved] AUC: {auc_p1:.5f}")

    # ==========================
    # Pseudo-Labeling (放宽阈值)
    # ==========================
    print("\n[Pseudo-Labeling] Generating Pseudo Labels...")
    # 伪标签生成逻辑
    test_pred_avg = np.mean(test_p1, axis=1)
    # 双阈值筛选：只要极其确定的样本
    high_conf_mask = (test_pred_avg > 0.90) | (test_pred_avg < 0.10)
    # 将测试集转化为训练集
    X_pseudo = X_test_raw[high_conf_mask].copy()
    y_pseudo = np.round(test_pred_avg[high_conf_mask]).astype(int)

    print(f"[Pseudo-Labeling] Added {len(X_pseudo)} high-confidence samples to Train.")
    final_oof = oof_p1
    final_test_preds = test_p1
    final_auc = auc_p1
    if len(X_pseudo) > 50:  # 只有样本够多才跑 Phase 2
        pseudo_groups = np.arange(10000, 10000 + len(X_pseudo))
        X_aug = pd.concat([X_raw, X_pseudo], axis=0).reset_index(drop=True)
        y_aug = pd.concat([pd.Series(y), pd.Series(y_pseudo)], axis=0).reset_index(drop=True)
        groups_aug = np.concatenate([groups, pseudo_groups])
        # Phase 2 训练
        oof_p2, test_p2, auc_p2 = run_phase(
            X_aug, y_aug, X_test_raw, groups_aug,
            n_splits, "Phase 2", seeds=SEEDS
        )

        final_oof = oof_p2[:len(y)]
        final_test_preds = test_p2
        final_auc = auc_p2
        print(f"\n[Result] Phase 2 improved AUC from {auc_p1:.5f} to {auc_p2:.5f}")
    else:
        print("\n[Result] Not enough pseudo samples. Using Phase 1 results.")

    # ==========================
    # 固定权重 Rank 融合（最终版本）
    # ==========================
    print("\n[Ensemble] Fixed Weight Rank Averaging (Final Version)...")

    # 固定权重：基于之前优化得到的最佳权重
    FIXED_WEIGHTS = np.array([0.00, 0.51, 0.49])  # [Base, GBDT, NN]
    print(f"Using fixed weights: Base={FIXED_WEIGHTS[0]:.2f}, "
          f"GBDT={FIXED_WEIGHTS[1]:.2f}, NN={FIXED_WEIGHTS[2]:.2f}")

    # 1. 计算 OOF 的 Rank (用于评估)
    oof_ranks = np.zeros_like(final_oof)
    for k in range(3):
        oof_ranks[:, k] = rankdata(final_oof[:, k]) / len(final_oof)

    # 固定权重 Rank 融合的 OOF AUC
    oof_rank_fused = np.dot(oof_ranks, FIXED_WEIGHTS)
    fixed_rank_auc = roc_auc_score(y, oof_rank_fused)
    print(f"Fixed Weight Rank AUC on OOF: {fixed_rank_auc:.5f}")
    print(f"Final Model AUC: {final_auc:.5f}")

    # 2. 计算测试集的 Rank
    test_ranks = np.zeros_like(final_test_preds)
    for k in range(3):
        test_ranks[:, k] = rankdata(final_test_preds[:, k]) / len(final_test_preds)

    # 应用固定权重融合
    final_pred_rank = np.dot(test_ranks, FIXED_WEIGHTS)
    final_pred_rank = np.clip(final_pred_rank, 0.0, 1.0)

    if test_ids is None:
        test_ids = np.arange(len(final_pred_rank))

    # 保存 Rank 融合结果
    sub_rank = pd.DataFrame({
        "id": test_ids.values if hasattr(test_ids, "values") else test_ids,
        "rainfall": final_pred_rank
    })
    sub_rank.to_csv("submission_final_rank.csv", index=False)
    print("Saved submission_final_rank.csv (Final Rank Fusion)")

    # ==========================
    # 最终概率平均作为对比
    # ==========================
    # 简单概率平均 (Base=0%, GBDT=40%, NN=60%)
    final_pred_prob = 0.4 * final_test_preds[:, 1] + 0.6 * final_test_preds[:, 2]

    # 计算概率平均的 OOF AUC
    oof_prob_fused = 0.4 * final_oof[:, 1] + 0.6 * final_oof[:, 2]
    prob_auc = roc_auc_score(y, oof_prob_fused)
    print(f"Final Probability Fusion AUC on OOF: {prob_auc:.5f}")

    sub_prob = pd.DataFrame({
        "id": test_ids.values if hasattr(test_ids, "values") else test_ids,
        "rainfall": final_pred_prob
    })
    sub_prob.to_csv("submission_final_prob.csv", index=False)
    print("Saved submission_final_prob.csv (40% GBDT + 60% NN)")

    # ==========================
    # 总结输出
    # ==========================
    print("\n" + "=" * 60)
    print("SUMMARY OF GENERATED FILES:")
    print("=" * 60)
    print("1. PHASE 1 (Pseudo-Label Version):")
    print("   - submission_pseudo_rank.csv: Phase 1 Rank fusion")
    print("   - submission_pseudo_prob.csv: Phase 1 Probability fusion")
    print(f"   - Phase 1 AUC: {auc_p1:.5f}")

    print("\n2. FINAL VERSION (with Pseudo-Labeling if available):")
    print("   - submission_final_rank.csv: Final Rank fusion")
    print("   - submission_final_prob.csv: Final Probability fusion")
    if len(X_pseudo) > 50:
        print(f"   - Final AUC: {final_auc:.5f} (improved from {auc_p1:.5f})")
    else:
        print(f"   - Final AUC: {final_auc:.5f} (same as Phase 1)")
    print("=" * 60)


if __name__ == "__main__":
    run_unified(n_splits=5)