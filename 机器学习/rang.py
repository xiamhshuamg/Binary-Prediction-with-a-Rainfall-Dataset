# weights_comparison.py
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入数据
from data_preprocessing import load_data


# ============================================================================
# 1. 基于AUC的加权融合策略
# ============================================================================
def auc_weighted_fusion(oof_preds: np.ndarray, y_true: np.ndarray,
                        test_preds: Optional[np.ndarray] = None) -> Dict:
    """
    基于AUC的加权融合策略
    """
    n_models = oof_preds.shape[1]

    # 计算每个模型的AUC
    model_aucs = []
    for i in range(n_models):
        auc = roc_auc_score(y_true, oof_preds[:, i])
        model_aucs.append(auc)

    # 转换为权重 (使用softmax-like函数)
    # 放大AUC差异：乘以10，然后softmax
    auc_array = np.array(model_aucs)
    exp_scores = np.exp(auc_array * 10)  # 乘以10放大差异
    weights = exp_scores / exp_scores.sum()

    # 计算融合后的预测
    oof_fused = np.dot(oof_preds, weights)
    fused_auc = roc_auc_score(y_true, oof_fused)

    result = {
        "method": "AUC Weighted",
        "weights": weights.tolist(),
        "model_aucs": model_aucs,
        "oof_auc": fused_auc,
        "weights_dict": dict(zip(["Baseline", "GBDT", "NN"], weights))
    }

    if test_preds is not None:
        test_fused = np.dot(test_preds, weights)
        result["test_preds"] = test_fused

    return result


# ============================================================================
# 2. 基于模型排名的权重
# ============================================================================
def rank_based_weights(oof_preds: np.ndarray, y_true: np.ndarray,
                       test_preds: Optional[np.ndarray] = None) -> Dict:
    """
    基于模型排名的权重分配
    """
    n_models = oof_preds.shape[1]

    # 计算每个模型的AUC
    model_aucs = []
    for i in range(n_models):
        auc = roc_auc_score(y_true, oof_preds[:, i])
        model_aucs.append(auc)

    # 根据AUC排名
    ranks = np.argsort(np.argsort(-np.array(model_aucs))) + 1  # 排名1最好

    # 权重分配：排名越靠前权重越高，使用指数衰减
    weights = np.exp(-np.array(ranks) / 2)  # 除以2控制衰减速度
    weights = weights / weights.sum()

    # 计算融合后的预测
    oof_fused = np.dot(oof_preds, weights)
    fused_auc = roc_auc_score(y_true, oof_fused)

    result = {
        "method": "Rank Based",
        "weights": weights.tolist(),
        "model_aucs": model_aucs,
        "ranks": ranks.tolist(),
        "oof_auc": fused_auc,
        "weights_dict": dict(zip(["Baseline", "GBDT", "NN"], weights))
    }

    if test_preds is not None:
        test_fused = np.dot(test_preds, weights)
        result["test_preds"] = test_fused

    return result


# ============================================================================
# 3. 相关性分析的权重
# ============================================================================
def correlation_aware_weights(oof_preds: np.ndarray, y_true: np.ndarray,
                              test_preds: Optional[np.ndarray] = None) -> Dict:
    """
    考虑模型相关性的权重分配
    """
    n_models = oof_preds.shape[1]

    # 1. 计算每个模型的AUC
    model_aucs = []
    for i in range(n_models):
        auc = roc_auc_score(y_true, oof_preds[:, i])
        model_aucs.append(auc)

    # 2. 计算模型间的相关性
    corr_matrix = np.corrcoef(oof_preds.T)

    # 3. 计算多样性分数 (相关性越低，多样性越高)
    diversity_scores = []
    for i in range(n_models):
        # 计算与其他模型的相关性平均值
        other_corrs = [corr_matrix[i, j] for j in range(n_models) if j != i]
        avg_corr = np.mean(other_corrs)
        diversity = 1 - avg_corr  # 相关性越低，多样性越高
        diversity_scores.append(diversity)

    # 4. 综合权重：70%基于AUC，30%基于多样性
    auc_array = np.array(model_aucs)
    div_array = np.array(diversity_scores)

    # 归一化
    auc_norm = auc_array / auc_array.sum()
    div_norm = div_array / div_array.sum()

    # 权重组合
    weights = 0.7 * auc_norm + 0.3 * div_norm
    weights = weights / weights.sum()

    # 计算融合后的预测
    oof_fused = np.dot(oof_preds, weights)
    fused_auc = roc_auc_score(y_true, oof_fused)

    result = {
        "method": "Correlation Aware",
        "weights": weights.tolist(),
        "model_aucs": model_aucs,
        "diversity_scores": diversity_scores,
        "correlation_matrix": corr_matrix.tolist(),
        "oof_auc": fused_auc,
        "weights_dict": dict(zip(["Baseline", "GBDT", "NN"], weights))
    }

    if test_preds is not None:
        test_fused = np.dot(test_preds, weights)
        result["test_preds"] = test_fused

    return result


# ============================================================================
# 4. 自适应权重
# ============================================================================
class AdaptiveWeightEnsemble:
    """自适应权重集成学习器"""

    def __init__(self, model_names: List[str], learning_rate: float = 0.1):
        self.model_names = model_names
        self.n_models = len(model_names)
        self.learning_rate = learning_rate
        self.weights = np.ones(self.n_models) / self.n_models
        self.weight_history = []
        self.fold_performance = []

    def update_weights(self, fold_aucs: np.ndarray):
        """根据fold表现更新权重"""
        # 将AUC转换为分数
        scores = np.array(fold_aucs)

        # 使用softmax获取目标权重
        exp_scores = np.exp(scores * 5)  # 放大差异
        target_weights = exp_scores / exp_scores.sum()

        # 平滑更新：新权重 = (1-lr)*旧权重 + lr*目标权重
        new_weights = (1 - self.learning_rate) * self.weights + self.learning_rate * target_weights

        # 归一化
        new_weights = new_weights / new_weights.sum()

        # 更新
        self.weights = new_weights
        self.weight_history.append(self.weights.copy())
        self.fold_performance.append(fold_aucs.copy())

        return self.weights

    def get_final_weights(self):
        """获取最终权重（基于历史权重的平均）"""
        if not self.weight_history:
            return self.weights

        # 取后50%的历史权重进行平均
        hist_len = len(self.weight_history)
        start_idx = max(0, hist_len // 2)

        avg_weights = np.zeros(self.n_models)
        for i in range(start_idx, hist_len):
            avg_weights += self.weight_history[i]

        return avg_weights / (hist_len - start_idx)


def adaptive_weights(oof_preds: np.ndarray, y_true: np.ndarray,
                     fold_indices: List[Tuple[np.ndarray, np.ndarray]],
                     test_preds: Optional[np.ndarray] = None) -> Dict:
    """
    自适应权重融合
    需要知道fold划分来模拟训练过程
    """
    n_models = oof_preds.shape[1]
    model_names = ["Baseline", "GBDT", "NN"]

    # 初始化自适应集成器
    ensemble = AdaptiveWeightEnsemble(model_names, learning_rate=0.1)

    # 模拟fold训练过程
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        # 获取当前fold的验证集预测
        fold_preds = oof_preds[val_idx]
        fold_true = y_true[val_idx]

        # 计算当前fold各模型的AUC
        fold_aucs = []
        for i in range(n_models):
            auc = roc_auc_score(fold_true, fold_preds[:, i])
            fold_aucs.append(auc)

        # 更新权重
        new_weights = ensemble.update_weights(fold_aucs)

        # 输出当前fold信息
        if fold < 3:  # 只显示前3个fold
            print(f"  Fold {fold + 1}: AUCs={[f'{a:.4f}' for a in fold_aucs]}, "
                  f"Weights={dict(zip(model_names, new_weights.round(3)))}")

    # 获取最终权重
    final_weights = ensemble.get_final_weights()

    # 计算融合后的预测
    oof_fused = np.dot(oof_preds, final_weights)
    fused_auc = roc_auc_score(y_true, oof_fused)

    result = {
        "method": "Adaptive Weights",
        "weights": final_weights.tolist(),
        "fold_performance": ensemble.fold_performance,
        "weight_history": ensemble.weight_history,
        "oof_auc": fused_auc,
        "weights_dict": dict(zip(model_names, final_weights))
    }

    if test_preds is not None:
        test_fused = np.dot(test_preds, final_weights)
        result["test_preds"] = test_fused

    return result


# ============================================================================
# 5. Rank融合（现有方法）
# ============================================================================
def rank_fusion(oof_preds: np.ndarray, y_true: np.ndarray,
                test_preds: Optional[np.ndarray] = None,
                n_trials: int = 20000) -> Dict:
    """
    Rank融合策略（与原pipeline相同）
    """
    n_models = oof_preds.shape[1]

    # 转换为Rank
    oof_ranks = np.zeros_like(oof_preds)
    for k in range(n_models):
        oof_ranks[:, k] = rankdata(oof_preds[:, k]) / len(oof_preds)

    # 随机搜索优化权重
    best_w = np.ones(n_models) / n_models
    best_auc = roc_auc_score(y_true, np.dot(oof_ranks, best_w))

    for i in range(n_trials):
        w = np.random.dirichlet(np.ones(n_models), size=1)[0]
        pred = np.dot(oof_ranks, w)
        auc_score = roc_auc_score(y_true, pred)
        if auc_score > best_auc:
            best_auc = auc_score
            best_w = w

    # 计算Rank融合结果
    oof_fused = np.dot(oof_ranks, best_w)
    fused_auc = best_auc

    result = {
        "method": "Rank Fusion",
        "weights": best_w.tolist(),
        "oof_auc": fused_auc,
        "weights_dict": dict(zip(["Baseline", "GBDT", "NN"], best_w))
    }

    if test_preds is not None:
        # 对测试集也进行Rank转换
        test_ranks = np.zeros_like(test_preds)
        for k in range(n_models):
            test_ranks[:, k] = rankdata(test_preds[:, k]) / len(test_preds)

        test_fused = np.dot(test_ranks, best_w)
        result["test_preds"] = test_fused

    return result


# ============================================================================
# 6. 简单基准方法
# ============================================================================
def simple_average(oof_preds: np.ndarray, y_true: np.ndarray,
                   test_preds: Optional[np.ndarray] = None) -> Dict:
    """
    简单平均融合（作为基准）
    """
    n_models = oof_preds.shape[1]
    weights = np.ones(n_models) / n_models

    oof_fused = np.mean(oof_preds, axis=1)
    fused_auc = roc_auc_score(y_true, oof_fused)

    result = {
        "method": "Simple Average",
        "weights": weights.tolist(),
        "oof_auc": fused_auc,
        "weights_dict": dict(zip(["Baseline", "GBDT", "NN"], weights))
    }

    if test_preds is not None:
        test_fused = np.mean(test_preds, axis=1)
        result["test_preds"] = test_fused

    return result


# ============================================================================
# 主对比函数
# ============================================================================
def compare_weight_strategies(oof_preds: np.ndarray, y_true: np.ndarray,
                              test_preds: Optional[np.ndarray] = None,
                              fold_indices: Optional[List] = None) -> pd.DataFrame:
    """
    对比所有权重融合策略
    """
    print("\n" + "=" * 60)
    print("权重融合策略对比分析")
    print("=" * 60)

    # 存储所有策略结果
    all_results = []

    # 1. 简单平均（基准）
    print("\n[1/6] 简单平均融合...")
    result_avg = simple_average(oof_preds, y_true, test_preds)
    all_results.append(result_avg)

    # 2. 基于AUC的加权融合
    print("[2/6] 基于AUC的加权融合...")
    result_auc = auc_weighted_fusion(oof_preds, y_true, test_preds)
    all_results.append(result_auc)

    # 3. 基于排名的权重
    print("[3/6] 基于排名的权重融合...")
    result_rank = rank_based_weights(oof_preds, y_true, test_preds)
    all_results.append(result_rank)

    # 4. 相关性分析的权重
    print("[4/6] 相关性分析的权重融合...")
    result_corr = correlation_aware_weights(oof_preds, y_true, test_preds)
    all_results.append(result_corr)

    # 5. Rank融合
    print("[5/6] Rank融合...")
    result_rank_fusion = rank_fusion(oof_preds, y_true, test_preds)
    all_results.append(result_rank_fusion)

    # 6. 自适应权重（需要fold信息）
    if fold_indices is not None:
        print("[6/6] 自适应权重融合...")
        result_adaptive = adaptive_weights(oof_preds, y_true, fold_indices, test_preds)
        all_results.append(result_adaptive)
    else:
        print("[6/6] 自适应权重融合跳过（需要fold信息）")

    # 创建结果DataFrame
    results_df = pd.DataFrame([{
        "Method": r["method"],
        "AUC": r["oof_auc"],
        "Baseline Weight": r["weights_dict"].get("Baseline", 0),
        "GBDT Weight": r["weights_dict"].get("GBDT", 0),
        "NN Weight": r["weights_dict"].get("NN", 0),
        "Weights": str(r["weights_dict"])
    } for r in all_results])

    # 按AUC排序
    results_df = results_df.sort_values("AUC", ascending=False).reset_index(drop=True)

    # 显示结果
    print("\n" + "=" * 60)
    print("权重融合策略对比结果（按AUC排序）")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # 计算提升幅度
    baseline_auc = results_df[results_df["Method"] == "Simple Average"]["AUC"].iloc[0]
    results_df["Improvement"] = results_df["AUC"] - baseline_auc

    print("\n相对简单平均的提升:")
    for _, row in results_df.iterrows():
        if row["Method"] != "Simple Average":
            print(f"{row['Method']}: +{row['Improvement']:.5f}")

    return results_df, all_results


# ============================================================================
# 可视化函数
# ============================================================================
def visualize_comparison(results_df: pd.DataFrame, all_results: List[Dict],
                         y_true: np.ndarray, oof_preds: np.ndarray):
    """
    可视化对比结果
    """
    print("\n生成可视化图表...")

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("权重融合策略对比分析", fontsize=16, fontweight='bold')

    # 1. AUC对比条形图
    ax1 = axes[0, 0]
    methods = results_df["Method"]
    aucs = results_df["AUC"]

    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    bars = ax1.bar(range(len(methods)), aucs, color=colors)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel("AUC")
    ax1.set_title("各融合策略的AUC对比")
    ax1.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, auc_val in zip(bars, aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{auc_val:.5f}', ha='center', va='bottom', fontsize=9)

    # 2. 权重分布热力图
    ax2 = axes[0, 1]
    weights_data = []
    model_names = ["Baseline", "GBDT", "NN"]

    for result in all_results:
        if "weights_dict" in result:
            weights = [result["weights_dict"].get(model, 0) for model in model_names]
            weights_data.append(weights)

    if weights_data:
        weights_matrix = np.array(weights_data).T  # 转置，使模型在行上
        im = ax2.imshow(weights_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(all_results)))
        ax2.set_xticklabels([r["method"] for r in all_results], rotation=45, ha='right')
        ax2.set_yticks(range(len(model_names)))
        ax2.set_yticklabels(model_names)
        ax2.set_title("各策略的权重分配")

        # 添加颜色条
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        # 添加数值文本
        for i in range(len(model_names)):
            for j in range(len(all_results)):
                ax2.text(j, i, f'{weights_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=9)

    # 3. 模型原始AUC
    ax3 = axes[0, 2]
    model_names = ["Baseline", "GBDT", "NN"]
    model_aucs = []

    for i in range(3):  # 三个模型
        auc = roc_auc_score(y_true, oof_preds[:, i])
        model_aucs.append(auc)

    colors_single = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars_single = ax3.bar(model_names, model_aucs, color=colors_single)
    ax3.set_ylabel("AUC")
    ax3.set_title("单一模型AUC表现")
    ax3.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, auc_val in zip(bars_single, model_aucs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{auc_val:.5f}', ha='center', va='bottom', fontsize=10)

    # 4. 预测分布对比
    ax4 = axes[1, 0]
    # 选择前3个策略的融合预测
    top_strategies = results_df.head(3)["Method"].tolist()

    for strategy_name in top_strategies:
        for result in all_results:
            if result["method"] == strategy_name and "test_preds" in result:
                sns.kdeplot(result["test_preds"], ax=ax4, label=strategy_name, alpha=0.7)

    ax4.set_xlabel("预测概率")
    ax4.set_ylabel("密度")
    ax4.set_title("Top 3策略的预测分布")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. 提升幅度对比
    ax5 = axes[1, 1]
    improvement_data = results_df[results_df["Method"] != "Simple Average"].copy()

    if not improvement_data.empty:
        methods_imp = improvement_data["Method"]
        improvements = improvement_data["Improvement"]

        colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
        bars_imp = ax5.bar(range(len(methods_imp)), improvements, color=colors_imp)

        ax5.set_xticks(range(len(methods_imp)))
        ax5.set_xticklabels(methods_imp, rotation=45, ha='right')
        ax5.set_ylabel("AUC提升")
        ax5.set_title("相对简单平均的提升幅度")
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax5.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, imp_val in zip(bars_imp, improvements):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2.,
                     height + (0.0002 if height >= 0 else -0.0005),
                     f'{imp_val:+.5f}', ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=9)

    # 6. 各模型预测相关性
    ax6 = axes[1, 2]
    corr_matrix = np.corrcoef(oof_preds.T)

    im_corr = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax6.set_xticks(range(3))
    ax6.set_xticklabels(model_names)
    ax6.set_yticks(range(3))
    ax6.set_yticklabels(model_names)
    ax6.set_title("模型预测相关性矩阵")

    # 添加相关系数文本
    for i in range(3):
        for j in range(3):
            ax6.text(j, i, f'{corr_matrix[i, j]:.3f}',
                     ha="center", va="center", color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                     fontsize=11, fontweight='bold')

    plt.colorbar(im_corr, ax=ax6, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("weights_comparison_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ 可视化图表已保存为 'weights_comparison_analysis.png'")

    return fig


# ============================================================================
# 保存结果函数
# ============================================================================
def save_comparison_results(all_results: List[Dict], test_ids: Optional[np.ndarray] = None):
    """
    保存所有融合策略的预测结果
    """
    print("\n保存各策略的预测结果...")

    os.makedirs("fusion_predictions", exist_ok=True)

    for result in all_results:
        if "test_preds" in result:
            method_name = result["method"].replace(" ", "_").lower()

            if test_ids is None:
                test_ids = np.arange(len(result["test_preds"]))

            df = pd.DataFrame({
                "id": test_ids,
                "rainfall": result["test_preds"]
            })

            filename = f"fusion_predictions/{method_name}_predictions.csv"
            df.to_csv(filename, index=False)
            print(f"✅ {result['method']}: 已保存到 {filename}")

    # 保存汇总信息
    summary_data = []
    for result in all_results:
        if "weights_dict" in result:
            summary_data.append({
                "Method": result["method"],
                "AUC": result.get("oof_auc", 0),
                "Baseline_Weight": result["weights_dict"].get("Baseline", 0),
                "GBDT_Weight": result["weights_dict"].get("GBDT", 0),
                "NN_Weight": result["weights_dict"].get("NN", 0)
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("AUC", ascending=False)
    summary_df.to_csv("fusion_predictions/summary.csv", index=False)
    print("✅ 汇总信息已保存到 'fusion_predictions/summary.csv'")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """
    主函数：加载数据，运行所有融合策略对比
    """
    print("开始权重融合策略对比分析...")
    print("=" * 60)

    # 1. 加载数据（简化版，使用数据预处理中的load_data）
    print("\n1. 加载数据...")
    try:
        # 加载数据（可以调整参数）
        X_train, y_train, X_valid, y_valid, X_test = load_data(
            valid_size=0.2,
            random_state=42,
            split_strategy="random",
            n_components=1,
            n_clusters=6
        )
        print(f"✓ 数据加载成功")
        print(f"  训练集: {X_train.shape}, 验证集: {X_valid.shape}, 测试集: {X_test.shape}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return

    # 2. 加载模型预测结果（这里需要实际运行模型或加载已有结果）
    print("\n2. 运行各模型获取预测...")
    print("注意：这部分需要运行三个模型，可能需要一些时间...")

    # 这里需要导入实际的模型函数并运行
    # 为了简化，我们先创建一个模拟的预测结果
    # 实际使用时，请用实际的模型运行结果替换这里

    try:
        from baseline.baseline_model import run_baseline
        from gbdt.gbdt_model import run_gbdt
        from nn_ensemble.nn_ensemble_model import run_nn_ensemble

        # 运行三个模型
        seed = 42

        print("  - 运行Baseline模型...")
        _, baseline_valid, baseline_test = run_baseline(
            X_train, y_train, X_valid, y_valid, X_test, seed=seed
        )

        print("  - 运行GBDT模型...")
        _, gbdt_valid, gbdt_test = run_gbdt(
            X_train, y_train, X_valid, y_valid, X_test, seed=seed
        )

        print("  - 运行NN Ensemble模型...")
        _, nn_valid, nn_test = run_nn_ensemble(
            X_train, y_train, X_valid, y_valid, X_test, seed=seed
        )

        # 组装预测矩阵
        oof_preds = np.column_stack([baseline_valid, gbdt_valid, nn_valid])
        test_preds = np.column_stack([baseline_test, gbdt_test, nn_test])

        print("✓ 模型预测获取成功")
        print(f"  OOF预测矩阵形状: {oof_preds.shape}")
        print(f"  测试预测矩阵形状: {test_preds.shape}")

    except Exception as e:
        print(f"✗ 模型运行失败: {e}")
        print("使用模拟数据进行演示...")
        # 创建模拟数据用于演示
        n_samples = len(y_valid)
        n_test = len(X_test)

        # 创建模拟的预测结果（基于你的AUC排名）
        np.random.seed(42)

        # 基于真实标签创建有噪声的预测
        baseline_valid = y_valid.values * 0.8 + np.random.normal(0, 0.15, n_samples)
        gbdt_valid = y_valid.values * 0.79 + np.random.normal(0, 0.16, n_samples)
        nn_valid = y_valid.values * 0.81 + np.random.normal(0, 0.14, n_samples)

        # 测试集预测（随机）
        baseline_test = np.random.uniform(0, 1, n_test)
        gbdt_test = np.random.uniform(0, 1, n_test)
        nn_test = np.random.uniform(0, 1, n_test)

        # 确保值在[0,1]范围内
        for arr in [baseline_valid, gbdt_valid, nn_valid, baseline_test, gbdt_test, nn_test]:
            np.clip(arr, 0, 1, out=arr)

        oof_preds = np.column_stack([baseline_valid, gbdt_valid, nn_valid])
        test_preds = np.column_stack([baseline_test, gbdt_test, nn_test])

        print("⚠ 使用模拟数据进行演示")

    # 3. 创建模拟的fold划分（用于自适应权重）
    print("\n3. 创建模拟fold划分...")
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(skf.split(oof_preds, y_valid))
    print(f"✓ 创建了 {len(fold_indices)} 个fold的划分")

    # 4. 运行所有融合策略对比
    print("\n4. 运行所有融合策略对比...")
    results_df, all_results = compare_weight_strategies(
        oof_preds, y_valid.values, test_preds, fold_indices
    )

    # 5. 可视化结果
    print("\n5. 生成可视化图表...")
    fig = visualize_comparison(results_df, all_results, y_valid.values, oof_preds)

    # 6. 保存结果
    print("\n6. 保存预测结果...")
    # 创建测试ID（如果没有的话）
    test_ids = np.arange(len(test_preds))
    save_comparison_results(all_results, test_ids)

    # 7. 输出最佳策略
    print("\n" + "=" * 60)
    print("🎯 最佳融合策略推荐")
    print("=" * 60)

    best_result = results_df.iloc[0]
    print(f"最佳策略: {best_result['Method']}")
    print(f"最佳AUC: {best_result['AUC']:.5f}")
    print(f"权重分配: {best_result['Weights']}")

    if best_result['Method'] != 'Simple Average':
        print(f"相对简单平均提升: +{best_result['Improvement']:.5f}")

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print("\n📁 生成的文件:")
    print("  - weights_comparison_analysis.png (可视化图表)")
    print("  - fusion_predictions/ (各策略的预测结果)")
    print("  - fusion_predictions/summary.csv (汇总信息)")

    return results_df, all_results


# ============================================================================
# 快速测试函数
# ============================================================================
def quick_test():
    """快速测试函数"""
    print("快速测试权重融合策略...")

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_test = 500

    # 创建真实标签
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # 创建三个模型的预测（有一定相关性但又有差异）
    # 模型1: 准确率最高
    pred1 = y_true * 0.85 + np.random.normal(0, 0.1, n_samples)

    # 模型2: 准确率中等
    pred2 = y_true * 0.80 + np.random.normal(0, 0.12, n_samples)

    # 模型3: 准确率较低
    pred3 = y_true * 0.75 + np.random.normal(0, 0.15, n_samples)

    # 测试集预测
    test_pred1 = np.random.uniform(0, 1, n_test)
    test_pred2 = np.random.uniform(0, 1, n_test)
    test_pred3 = np.random.uniform(0, 1, n_test)

    # 确保值在[0,1]范围内
    for arr in [pred1, pred2, pred3, test_pred1, test_pred2, test_pred3]:
        np.clip(arr, 0, 1, out=arr)

    oof_preds = np.column_stack([pred1, pred2, pred3])
    test_preds = np.column_stack([test_pred1, test_pred2, test_pred3])

    # 创建fold划分
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(skf.split(oof_preds, y_true))

    # 运行对比
    results_df, all_results = compare_weight_strategies(
        oof_preds, y_true, test_preds, fold_indices
    )

    return results_df, all_results


# ============================================================================
# 执行
# ============================================================================
if __name__ == "__main__":
    # 运行完整分析
    results_df, all_results = main()

    # 或者运行快速测试
    # results_df, all_results = quick_test()