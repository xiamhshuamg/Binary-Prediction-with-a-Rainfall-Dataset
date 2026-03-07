from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
DEFAULT_TEST_PATH = os.path.join(DATA_DIR, "test.csv")
DEFAULT_TARGET = "rainfall"


def add_global_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """全局时序特征：滚动统计与滞后"""
    df = df.copy()
    if "day" in df.columns:
        df = df.sort_values("day").reset_index(drop=True)

    # 1. Log 变换
    for col in ["windspeed", "pressure", "humidity", "cloud", "sunshine"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # 2. 时序特征
    # 窗口选择：1天(昨天), 3天(短期趋势), 7天(一周趋势)
    rolling_cols = ["temperature", "pressure", "humidity", "windspeed"]
    windows = [1, 3, 7]

    for col in rolling_cols:
        if col not in df.columns: continue
        for w in windows:
            if w == 1:
                # 昨天的直接值，这是最强的特征之一
                df[f"{col}_lag_1"] = df[col].shift(1)
                df[f"{col}_diff_1"] = df[col].diff(1)
            else:
                r = df[col].rolling(window=w, min_periods=1)
                df[f"{col}_mean_{w}d"] = r.mean()
                df[f"{col}_std_{w}d"] = r.std()
                # 趋势：今天相对于过去几天的变化
                df[f"{col}_trend_{w}d"] = df[col] - df[f"{col}_mean_{w}d"]

    df = df.bfill().ffill().fillna(0)
    return df


def read_raw_data(
        train_path: str = DEFAULT_TRAIN_PATH,
        test_path: str = DEFAULT_TEST_PATH,
        target: str = DEFAULT_TARGET,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练文件: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试文件: {test_path}")

    train = pd.read_csv(train_path).rename(columns={"temparature": "temperature"})
    test = pd.read_csv(test_path).rename(columns={"temparature": "temperature"})

    test_ids = test["id"].copy() if "id" in test.columns else None

    if "id" in train.columns: train = train.drop(columns=["id"])
    if "id" in test.columns: test = test.drop(columns=["id"])

    # --- 核心：先拼接，做全局特征，再分离 ---
    train['is_train'] = 1
    test['is_train'] = 0
    test[target] = np.nan

    full_df = pd.concat([train, test], axis=0, ignore_index=True)
    full_df = add_global_time_features(full_df)

    # 分离
    train_sorted = full_df[full_df['is_train'] == 1].copy()
    test_sorted = full_df[full_df['is_train'] == 0].copy()

    y = train_sorted[target].astype(int)
    X_train = train_sorted.drop(columns=[target, 'is_train']).reset_index(drop=True)
    X_test = test_sorted.drop(columns=[target, 'is_train']).reset_index(drop=True)

    return X_train, y, X_test, test_ids


def _fit_extreme_thresholds(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    thresholds = {}
    for col in ["temperature", "humidity", "pressure", "windspeed"]:
        if col in train_df.columns:
            thresholds[col] = (float(train_df[col].quantile(0.01)), float(train_df[col].quantile(0.99)))
    return thresholds


def engineer_features_local(
        df: pd.DataFrame,
        extreme_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """局部特征工程：物理交互"""
    enhanced = df.copy()

    # 1. 风向循环编码 (将 359度和1度 连起来)
    if "winddirection" in enhanced.columns:
        wd_rad = np.deg2rad(enhanced["winddirection"])
        enhanced["wind_sin"] = np.sin(wd_rad)
        enhanced["wind_cos"] = np.cos(wd_rad)

    # 2. 气象物理特征
    # 露点差 (Dewpoint Depression): 越小越容易结露/下雨
    if "temperature" in enhanced.columns and "dewpoint" in enhanced.columns:
        enhanced["dewpoint_depression"] = enhanced["temperature"] - enhanced["dewpoint"]

    # 湿压指数 (简单的自定义指数: 高湿+低压 = 雨)
    if "humidity" in enhanced.columns and "pressure" in enhanced.columns:
        # 归一化一下 roughly: pressure ~1000, humidity ~100
        # 用 Log 后的值交互更稳定
        if "log_humidity" in enhanced.columns and "log_pressure" in enhanced.columns:
            enhanced["moisture_pressure_interaction"] = enhanced["log_humidity"] / (enhanced["log_pressure"] + 1)

    # 日照云量比
    if "sunshine" in enhanced.columns and "cloud" in enhanced.columns:
        enhanced["sun_cloud_ratio"] = enhanced["sunshine"] / (enhanced["cloud"] + 0.1)

    # 3. Z-Score 相对值
    for col in ["temperature", "humidity", "pressure"]:
        if col in enhanced.columns:
            mean_v = enhanced[col].mean()
            std_v = enhanced[col].std()
            if std_v > 0:
                enhanced[f"z_{col}"] = (enhanced[col] - mean_v) / std_v

    return enhanced


def _drop_constant_columns(df: pd.DataFrame) -> List[str]:
    nunique = df.nunique(dropna=False)
    return nunique[nunique <= 1].index.tolist()


@dataclass
class Preprocessor:
    n_components: int = 1
    n_clusters: int = 6
    base_cols_: Optional[List[str]] = None
    imputer_: Optional[SimpleImputer] = None
    indicator_cols_used_: Optional[List[str]] = None
    svd_: Optional[TruncatedSVD] = None
    svd_k_: int = 0
    extreme_thresholds_: Optional[Dict] = None
    kmeans_: Optional[KMeans] = None
    kmeans_scaler_: Optional[StandardScaler] = None
    scaler_: Optional[StandardScaler] = None
    drop_cols_: Optional[List[str]] = None
    feature_columns_: Optional[List[str]] = None

    def fit(self, X_train_raw: pd.DataFrame) -> "Preprocessor":
        self.base_cols_ = list(X_train_raw.columns)
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp_values = self.imputer_.fit_transform(X_train_raw[self.base_cols_])
        X_imp = pd.DataFrame(X_imp_values, columns=self.base_cols_, index=X_train_raw.index)

        # SVD for missing patterns
        indicator_cols = []
        for c in self.base_cols_:
            if X_train_raw[c].isna().any():
                ind = f"{c}_missing"
                X_imp[ind] = X_train_raw[c].isna().astype(int)
                indicator_cols.append(ind)
        self.indicator_cols_used_ = [c for c in indicator_cols if X_imp[c].nunique() > 1]

        if self.indicator_cols_used_ and len(self.indicator_cols_used_) > 1 and self.n_components > 0:
            M = X_imp[self.indicator_cols_used_].values
            self.svd_k_ = min(self.n_components, len(self.indicator_cols_used_))
            self.svd_ = TruncatedSVD(n_components=self.svd_k_, random_state=42)
            Z = self.svd_.fit_transform(M)
            for i in range(self.svd_k_):
                X_imp[f"missing_svd_{i}"] = Z[:, i]
            X_imp.drop(columns=self.indicator_cols_used_, inplace=True)
        else:
            if indicator_cols: X_imp.drop(columns=indicator_cols, inplace=True)

        self.extreme_thresholds_ = _fit_extreme_thresholds(X_imp)
        X_fe = engineer_features_local(X_imp, self.extreme_thresholds_)

        # KMeans (加入 humidity 和 pressure, 捕捉暴雨模式)
        cluster_cols = [c for c in ["temperature", "humidity", "pressure", "windspeed", "cloud"] if c in X_fe.columns]
        if cluster_cols and self.n_clusters > 1:
            self.kmeans_scaler_ = StandardScaler()
            X_cluster_std = self.kmeans_scaler_.fit_transform(X_fe[cluster_cols])
            self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.kmeans_.fit(X_cluster_std)
            dist = self.kmeans_.transform(X_cluster_std)
            for i in range(self.n_clusters):
                X_fe[f"cluster_dist_{i}"] = dist[:, i]

        self.scaler_ = StandardScaler()
        X_scaled_values = self.scaler_.fit_transform(X_fe)
        X_scaled = pd.DataFrame(X_scaled_values, columns=X_fe.columns, index=X_fe.index)

        self.drop_cols_ = _drop_constant_columns(X_scaled)
        if self.drop_cols_: X_scaled.drop(columns=self.drop_cols_, inplace=True)
        self.feature_columns_ = list(X_scaled.columns)
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        if self.base_cols_ is None: raise RuntimeError("Fitted first.")
        X_align = X_raw.copy()
        for c in self.base_cols_:
            if c not in X_align.columns: X_align[c] = np.nan
        X_align = X_align[self.base_cols_]

        X_imp_values = self.imputer_.transform(X_align)
        X_imp = pd.DataFrame(X_imp_values, columns=self.base_cols_, index=X_raw.index)

        if self.indicator_cols_used_:
            for col_orig in [c.replace("_missing", "") for c in self.indicator_cols_used_]:
                if col_orig in X_raw.columns:
                    X_imp[f"{col_orig}_missing"] = X_raw[col_orig].isna().astype(int)
                else:
                    X_imp[f"{col_orig}_missing"] = 0
            if self.svd_ and self.svd_k_ > 0:
                M = X_imp[self.indicator_cols_used_].values
                Z = self.svd_.transform(M)
                for i in range(self.svd_k_):
                    X_imp[f"missing_svd_{i}"] = Z[:, i]
                X_imp.drop(columns=self.indicator_cols_used_, inplace=True)
            else:
                X_imp.drop(columns=self.indicator_cols_used_, inplace=True, errors='ignore')

        X_fe = engineer_features_local(X_imp, self.extreme_thresholds_)

        if self.kmeans_ and self.kmeans_scaler_:
            cluster_cols = [c for c in ["temperature", "humidity", "pressure", "windspeed", "cloud"] if
                            c in X_fe.columns]
            if cluster_cols:
                X_cluster_std = self.kmeans_scaler_.transform(X_fe[cluster_cols])
                dist = self.kmeans_.transform(X_cluster_std)
                for i in range(self.n_clusters):
                    X_fe[f"cluster_dist_{i}"] = dist[:, i]

        X_scaled_values = self.scaler_.transform(X_fe)
        X_scaled = pd.DataFrame(X_scaled_values, columns=X_fe.columns, index=X_fe.index)
        if self.drop_cols_: X_scaled.drop(columns=self.drop_cols_, inplace=True, errors="ignore")
        for c in self.feature_columns_:
            if c not in X_scaled.columns: X_scaled[c] = 0.0
        return X_scaled[self.feature_columns_]


# 完整的 load_data 函数，支持本地测试
def load_data(
        train_path: str = DEFAULT_TRAIN_PATH,
        test_path: str = DEFAULT_TEST_PATH,
        target: str = DEFAULT_TARGET,
        valid_size: float = 0.2,
        random_state: int = 42,
        split_strategy: str = "random",  # "random" or "time"
        n_components: int = 1,
        n_clusters: int = 6,  # 新增参数
):
    """
    便捷加载函数。
    注意：在统一管线(unified_pipeline)中，我们通常不直接用这个函数，
    而是手动调用 Preprocessor 以实现按折交叉验证。
    """
    # 1. 读取（此时已经包含了全局时序特征）
    X_raw, y, X_test_raw, _ = read_raw_data(train_path, test_path, target=target)

    # 2. 简单切分用于测试
    if split_strategy == "time" and "day" in X_raw.columns:
        day = X_raw["day"].values
        order = np.argsort(day)
        cut = int(len(order) * (1 - valid_size))
        tr_idx, va_idx = order[:cut], order[cut:]
    else:
        # Stratified Split
        tr_idx, va_idx = train_test_split(
            np.arange(len(y)), test_size=valid_size, random_state=random_state, stratify=y
        )

    X_tr_raw = X_raw.iloc[tr_idx].copy()
    y_train = y.iloc[tr_idx].copy()
    X_va_raw = X_raw.iloc[va_idx].copy()
    y_valid = y.iloc[va_idx].copy()

    # 3. Fit on train
    pre = Preprocessor(n_components=n_components, n_clusters=n_clusters).fit(X_tr_raw)

    # 4. Transform
    X_train = pre.transform(X_tr_raw)
    X_valid = pre.transform(X_va_raw)
    X_test = pre.transform(X_test_raw)

    return X_train, y_train, X_valid, y_valid, X_test


if __name__ == "__main__":
    print("Testing Preprocessor...")
    # 这里现在可以正常运行了
    X_tr, y_tr, X_va, y_va, X_te = load_data()
    print(f"Train shape: {X_tr.shape}, Valid shape: {X_va.shape}, Test shape: {X_te.shape}")
    print("Test passed.")