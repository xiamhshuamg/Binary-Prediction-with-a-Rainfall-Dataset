# unified_pipeline_simple.py
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


def run_phase_only(X, y, X_test, groups, n_splits=5, phase_name="Phase", seeds=[42], num_phases=3):
    """
    运行多个Phase（多轮评估），不使用伪标签，不进行融合
    num_phases: 运行多少个Phase（轮次）
    """
    print(f"\n=== 开始运行 {num_phases} 个 Phase ===")

    # 保存每个Phase的结果
    all_oof_preds = []
    all_test_preds = []
    all_aucs = []

    for phase_idx in range(1, num_phases + 1):
        print(f"\n{'=' * 50}")
        print(f"Phase {phase_idx} 开始 (训练样本={len(X)}, Seeds={len(seeds)})")
        print(f"{'=' * 50}")

        # 容器：累加所有 Seed 的结果
        oof_preds_total = np.zeros((len(y), 3), dtype=float)
        test_preds_total = np.zeros((len(X_test), 3), dtype=float)

        for seed_idx, seed in enumerate(seeds, 1):
            print(f"  > Seed {seed_idx}/{len(seeds)}: {seed}")

            # 使用 StratifiedGroupKFold
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

            # 单次 Seed 的容器
            oof_preds_seed = np.zeros((len(y), 3), dtype=float)
            test_preds_seed = np.zeros((len(X_test), 3), dtype=float)

            for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups), 1):
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

                if fold_idx % 2 == 0 or fold_idx == n_splits:
                    print(f"    Fold {fold_idx}/{n_splits} 完成")

            # Seed 内部 normalize test preds (除以 n_splits)
            test_preds_seed /= n_splits

            # 累加到 Total
            oof_preds_total += oof_preds_seed
            test_preds_total += test_preds_seed

            # 计算当前 Seed 的 AUC
            avg_seed_p = np.mean(oof_preds_seed, axis=1)
            seed_auc = roc_auc_score(y, avg_seed_p)
            print(f"  > Seed {seed} AUC: {seed_auc:.5f}")

        # 取所有 Seeds 的平均
        oof_preds_final = oof_preds_total / len(seeds)
        test_preds_final = test_preds_total / len(seeds)

        # 计算最终AUC
        avg_p_final = np.mean(oof_preds_final, axis=1)
        final_auc = roc_auc_score(y, avg_p_final)

        # 保存这个Phase的结果
        all_oof_preds.append(oof_preds_final)
        all_test_preds.append(test_preds_final)
        all_aucs.append(final_auc)

        print(f"\n[Phase {phase_idx}] 完成 - {len(seeds)}个Seeds的平均 AUC: {final_auc:.5f}")

        # 计算每个模型的独立AUC
        print("\n  各个模型在OOF上的性能:")
        for i, model_name in enumerate(["Baseline", "GBDT", "NN Ensemble"]):
            model_auc = roc_auc_score(y, oof_preds_final[:, i])
            print(f"    {model_name}: AUC = {model_auc:.5f}")

    return all_oof_preds, all_test_preds, all_aucs


def run_simple_pipeline(
        n_splits: int = 5,
        num_phases: int = 3,
        output_path: str = "submission_phase_only.csv",
):
    """
    简化的Pipeline：只运行多个Phase，不使用伪标签，不进行融合
    """
    # 读取数据
    X_raw, y, X_test_raw, test_ids = read_raw_data()
    y = y.values.astype(int)
    groups = get_day_col(X_raw)

    # 使用多个Seed
    SEEDS = [42, 2023, 1024, 12345, 67890][:3]  # 最多用5个，但默认用前3个

    # 运行多个Phase
    all_oof_preds, all_test_preds, all_aucs = run_phase_only(
        X_raw, pd.Series(y), X_test_raw, groups,
        n_splits=n_splits, seeds=SEEDS, num_phases=num_phases
    )

    # 分析不同Phase的结果
    print(f"\n{'=' * 60}")
    print(f"所有 {num_phases} 个 Phase 的结果汇总")
    print(f"{'=' * 60}")

    for i, auc in enumerate(all_aucs, 1):
        print(f"Phase {i}: AUC = {auc:.5f}")

    # 选择最佳Phase
    best_phase_idx = np.argmax(all_aucs)
    print(f"\n最佳Phase是 Phase {best_phase_idx + 1}, AUC = {all_aucs[best_phase_idx]:.5f}")

    # 使用最佳Phase的结果
    best_oof_preds = all_oof_preds[best_phase_idx]
    best_test_preds = all_test_preds[best_phase_idx]

    # 保存每个模型的预测结果（不融合）
    print(f"\n{'=' * 60}")
    print("保存各个模型的预测结果（不融合）")
    print(f"{'=' * 60}")

    model_names = ["baseline", "gbdt", "nn_ensemble"]

    for i, (model_name, model_auc) in enumerate(zip(model_names,
                                                    [roc_auc_score(y, best_oof_preds[:, j])
                                                     for j in range(3)])):
        # 生成测试集预测
        test_pred_single = best_test_preds[:, i]

        # 保存为CSV
        submission = pd.DataFrame({
            "id": test_ids.values if hasattr(test_ids, "values") else test_ids,
            "rainfall": test_pred_single
        })

        filename = f"submission_{model_name}_phase{best_phase_idx + 1}.csv"
        submission.to_csv(filename, index=False)
        print(f"  保存 {model_name} 预测结果到 {filename} (OOF AUC: {model_auc:.5f})")

    # 另外保存一个简单平均的结果（可选）
    test_pred_avg = np.mean(best_test_preds, axis=1)
    avg_auc = roc_auc_score(y, np.mean(best_oof_preds, axis=1))

    submission_avg = pd.DataFrame({
        "id": test_ids.values if hasattr(test_ids, "values") else test_ids,
        "rainfall": test_pred_avg
    })

    submission_avg.to_csv(output_path, index=False)
    print(f"\n  保存简单平均结果到 {output_path} (OOF AUC: {avg_auc:.5f})")

    # 评估各个Phase在OOF上的表现
    print(f"\n{'=' * 60}")
    print("各Phase的OOF评估结果")
    print(f"{'=' * 60}")

    for phase_idx, (oof_preds, phase_auc) in enumerate(zip(all_oof_preds, all_aucs), 1):
        print(f"\nPhase {phase_idx} (AUC: {phase_auc:.5f}):")
        for i, model_name in enumerate(["Baseline", "GBDT", "NN Ensemble"]):
            model_auc = roc_auc_score(y, oof_preds[:, i])
            print(f"  {model_name}: AUC = {model_auc:.5f}")

    return all_oof_preds, all_test_preds, all_aucs


if __name__ == "__main__":
    # 可以调整参数：n_splits=5, num_phases=3
    run_simple_pipeline(n_splits=5, num_phases=3)