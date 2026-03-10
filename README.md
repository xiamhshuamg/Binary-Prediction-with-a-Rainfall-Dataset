# 基于降雨数据集的二分类预测

本项目是**机器学习技术与实践**课程期末大作业，旨在针对 Kaggle 降雨预测竞赛任务，构建一个端到端、可复现、高稳健的短临降水发生概率预测系统。项目围绕气象数据的时序性、类别不平衡、非线性交互与泛化波动等挑战，设计并实现了统一预处理、多模型并行、伪标签增广、Rank融合等策略，最终在按 day 分组的分层交叉验证下取得 **OOF AUC = 0.90291** 的稳定表现，并产出符合竞赛格式的提交文件。

## 🧭 项目简介

降雨预测是气象服务与城市治理中的关键任务，其核心在于利用温度、湿度、气压、云量等多维气象观测值，判断某日是否发生降雨。本项目以 Kaggle 竞赛数据集为基础，将任务定义为**二分类概率预测**，并重点解决：

- **时序依赖**：相邻日期的气象分布高度相关，需防止验证集时间泄漏。
- **特征交互**：降雨常由多变量耦合触发，需构造趋势、波动、交互特征。
- **类别不平衡**：有雨/无雨样本比例不均，需采用 AUC 等阈值无关指标。
- **泛化波动**：小样本下随机种子与划分方式易导致结果不稳定，需引入多 seed 集成与融合。

项目最终交付完整的训练—验证—推理—提交闭环，所有代码模块化、可复现，并保留了关键中间产物（OOF 预测、种子日志、融合对比结果）供分析与复盘。

## ✨ 主要成果与特点

- ✅ **端到端可复现管线**：从数据读取、特征构造、交叉验证训练到提交文件生成，统一由 `unified_pipeline.py` 调度，支持多 seed 重复实验与结果平均。
- ✅ **严谨的验证策略**：采用 **StratifiedGroupKFold** 以 day 为分组单位，确保同一天样本不跨折拆分，同时保持各折正负比例接近全局分布，有效避免时序泄漏。
- ✅ **多模型互补架构**：构建三路并行分支：
  - **Baseline**：逻辑回归 + 随机森林（兜底、可解释对照）
  - **GBDT**：XGBoost、LightGBM、CatBoost 并行（主力非线性拟合）
  - **轻量集成**：KNN、ExtraTrees、简化 MLP（提供差异化偏差）
- ✅ **可控伪标签增广**：仅在 Phase1 高置信预测（p > 0.90 或 p < 0.10）且数量超过 50 时启动 Phase2，在控制噪声前提下提升泛化能力。
- ✅ **稳健的融合策略**：采用 **Rank Averaging** 对不同模型输出进行排序归一化后加权融合，降低概率尺度差异影响，最终提交文件由固定权重（GBDT:0.51，轻量集成:0.49）生成。

## 🛠️ 技术栈

- **语言**：Python 3.9+
- **数据处理**：pandas, numpy, scikit-learn
- **机器学习**：
  - XGBoost, LightGBM, CatBoost（GBDT 分支）
  - scikit-learn 实现逻辑回归、随机森林、KNN、ExtraTrees
  - 简化 MLP（基于 numpy 手写）
- **验证工具**：自定义 `StratifiedGroupKFold`，`roc_auc_score`
- **环境管理**：requirements.txt / pip

## 🧠 方法概述

### 数据预处理与特征工程

- **统一口径**：先将 train/test 合并，生成全局时序特征后再拆分，保证特征分布一致。
- **缺失处理**：填补（中位数）+ 缺失指示器，保留缺失模式信息；对风向进行循环编码。
- **时序特征**：基于 day 排序构造滞后、差分、滚动均值/方差（3/7 天）、趋势项。
- **交互特征**：温度与露点差、湿度与气压比值、日照与云量比值等物理组合。
- **聚类特征**：在关键气象子空间使用 KMeans 计算样本到中心距离，增强可分性。

### 验证设计

- 使用 **StratifiedGroupKFold**，n_splits=5，group=day。
- 每个 seed 独立运行完整 5 折交叉验证，输出 OOF 预测并计算 AUC。
- 多 seed 平均（seeds=[42,2023,1024]）降低随机波动，得到最终 OOF AUC。

### 模型分支与训练

- **Baseline**：`run_baseline()` 包含 numpy 逻辑回归 + LightGBM rf 风格森林（降级为极简森林兜底）。
- **GBDT**：`run_gbdt()` 并行训练 XGB/LGB/CB，折内输出三模型均值。
- **轻量集成**：`run_nn_ensemble()` 组合 KNN、ExtraTrees（近似）、简化 MLP，输出均值概率。

### 伪标签策略

- Phase1 得到测试集平均预测 `test_pred_avg`。
- 筛选高置信样本（>0.90 或 <0.10），若数量 >50 则为其生成伪标签并拼接至训练集（分配新的 group 编号）。
- 调用同一 `run_phase()` 进行 Phase2 训练，再次输出 OOF 与测试预测。

### 融合与提交

- 对三路分支输出执行 **Rank Averaging**：`scipy.stats.rankdata` 归一化后按固定权重加权平均。
- 最终概率经 clip(0,1) 后写入 `submission_rank_fixed.csv`，同时保留常规均值融合版本 `submission.csv` 供对比。

## 📊 实验结果

| 阶段 | 策略 | OOF AUC（多 seed 平均） |
|------|------|--------------------------|
| Phase1 | 基础三路融合 | 0.89054 |
| Phase2 | 伪标签增广后融合 | **0.90291** |
| 融合对比 | 概率均值 vs Rank 融合 | Rank 融合更稳定，选定权重 [GBDT=0.51, NN=0.49] |

- 伪标签筛选：高置信样本 112 条（>50），触发 Phase2，带来 +0.01237 提升。
- 最终提交文件经 Rank 融合生成，在所有 seed 上波动 < 0.003，具备良好复现性。

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 安装依赖：`pip install -r requirements.txt`

### 运行全流程
```bash
python unified_pipeline.py
```
程序会自动：
1. 读取 `train.csv`、`test.csv`
2. 执行统一特征工程
3. 运行 Phase1 多 seed 训练
4. 根据条件决定是否执行 Phase2
5. 生成融合提交文件 `submission_rank_fixed.csv`
6. 打印各 seed AUC 及最终平均 AUC

### 自定义配置
可在脚本头部修改 `seeds`、`n_splits`、伪标签阈值、融合权重等参数。

## 📁 项目结构

```
.
├── unified_pipeline.py          # 主流程脚本
├── data_preprocessing.py        # 预处理器（缺失、标准化、聚类、特征工程）
├── ml_tools.py                  # StratifiedGroupKFold、AUC 等工具
├── baseline_model.py            # Baseline 分支（LR + 森林兜底）
├── gbdt_model.py                # GBDT 分支（XGB/LGB/CB）
├── nn_ensemble_model.py         # 轻量集成分支（KNN/ExtraTrees/简化MLP）
├── requirements.txt             # 依赖列表
├── data/                        # 存放 train.csv, test.csv（需自行放置）
└── README.md
```



## 📄 许可证

本项目为课程实践作品，仅供学习交流使用。

---

> **项目状态**：已完成核心功能闭环，实验结果可复现，具备良好扩展性。后续可引入自动超参搜索、多源数据融合、业务风险分层等方向迭代。
