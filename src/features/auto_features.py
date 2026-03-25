"""
auto_features.py
----------------
用 tsfresh 對每支球隊的近期比賽序列自動生成統計特徵，
補充手工特徵沒有覆蓋到的時序統計量。

流程：
  1. 從 features.csv 載入手工特徵
  2. 為每支球隊建立近 20 場得分時序
  3. tsfresh 自動抽取特徵（autocorrelation, entropy 等）
  4. 特徵選擇（去除低變異、高相關）
  5. 合併回主 DataFrame，輸出 features_full.csv

使用方式：
  python src/features/auto_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

# ── 設定 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

FEAT_DIR = Path(__file__).parent.parent.parent / CFG["data"]["features_dir"]
SEQ_LEN  = 15   # 取近 15 場作為時序


# ── 建立時序資料 ──────────────────────────────────────────────

def build_team_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    把每場比賽轉成「以球隊為單位的時序資料」，
    格式符合 tsfresh 要求：
      id         : "{game_id}_{home/away}"
      time       : 0..SEQ_LEN-1（最近的在最後）
      scored     : 該場得分
      allowed    : 該場失分
      total      : 該場總得分
    """
    rows = []

    home_view = df[["game_id", "game_date", "home_team",
                    "home_total", "away_total", "total_score"]].copy()
    home_view.rename(columns={"home_team": "team",
                               "home_total": "scored",
                               "away_total": "allowed"}, inplace=True)
    home_view["side"] = "home"

    away_view = df[["game_id", "game_date", "away_team",
                    "away_total", "home_total", "total_score"]].copy()
    away_view.rename(columns={"away_team": "team",
                               "away_total": "scored",
                               "home_total": "allowed"}, inplace=True)
    away_view["side"] = "away"

    long = pd.concat([home_view, away_view]).sort_values("game_date")

    # 對每場比賽的每支球隊，取「賽前」近 SEQ_LEN 場歷史
    all_games_sorted = df.sort_values("game_date")

    for _, game_row in all_games_sorted.iterrows():
        gid   = game_row["game_id"]
        gdate = game_row["game_date"]

        for side in ["home", "away"]:
            team = game_row[f"{side}_team"]

            # 取該隊在這場比賽之前的近 SEQ_LEN 場
            past = long[(long["team"] == team) &
                        (long["game_date"] < gdate)].tail(SEQ_LEN).reset_index(drop=True)

            if len(past) < 5:
                continue  # 歷史太少，跳過

            for t, (_, pr) in enumerate(past.iterrows()):
                rows.append({
                    "id":      f"{gid}__{side}",
                    "time":    t,
                    "scored":  pr["scored"],
                    "allowed": pr["allowed"],
                    "total":   pr["total_score"],
                })

    ts_df = pd.DataFrame(rows)
    log.info(f"Built timeseries: {ts_df['id'].nunique()} series, "
             f"{len(ts_df)} rows")
    return ts_df


# ── tsfresh 特徵抽取 ──────────────────────────────────────────

def extract_tsfresh_features(ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    用 MinimalFCParameters 抽取輕量特徵集（約 20 種統計量），
    避免全量抽取（700+ 特徵）造成過擬合。

    MinimalFCParameters 包含：
      mean, median, min, max, std,
      sum, length, variance, skewness, kurtosis,
      autocorrelation(lag=1), number_of_peaks, ...
    """
    log.info("Extracting tsfresh features (this may take a few minutes)...")

    features = extract_features(
        ts_df,
        column_id="id",
        column_sort="time",
        column_value=None,          # 自動偵測數值欄
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=4,                   # 平行處理
        disable_progressbar=False,
    )

    # 填補 NaN
    impute(features)

    log.info(f"Extracted {features.shape[1]} tsfresh features "
             f"for {features.shape[0]} series")
    return features


# ── 特徵選擇 ──────────────────────────────────────────────────

def select_features(features: pd.DataFrame,
                    variance_threshold: float = 0.01,
                    corr_threshold: float = 0.95) -> pd.DataFrame:
    """
    簡單的特徵篩選：
      1. 移除近乎零變異的特徵
      2. 移除高度相關的特徵（保留一個）
    """
    before = features.shape[1]

    # 1. 低變異過濾
    variances = features.var()
    features  = features.loc[:, variances > variance_threshold]
    log.info(f"After variance filter: {features.shape[1]} / {before}")

    # 2. 高相關過濾
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    features = features.drop(columns=to_drop)
    log.info(f"After correlation filter: {features.shape[1]} features remaining")

    return features


# ── 合併回主 DataFrame ────────────────────────────────────────

def merge_auto_features(df: pd.DataFrame,
                         auto_feats: pd.DataFrame) -> pd.DataFrame:
    """
    把 tsfresh 特徵（home / away 兩個 series）合併回比賽 DataFrame。
    """
    # tsfresh index 格式："{game_id}__home" / "{game_id}__away"
    home_feats = auto_feats[auto_feats.index.str.endswith("__home")].copy()
    away_feats = auto_feats[auto_feats.index.str.endswith("__away")].copy()

    home_feats.index = home_feats.index.str.replace("__home", "")
    away_feats.index = away_feats.index.str.replace("__away", "")

    home_feats = home_feats.add_prefix("h_auto_")
    away_feats = away_feats.add_prefix("a_auto_")

    # 統一型別：game_id 轉成字串再 join
    df = df.copy()
    df["game_id_str"] = df["game_id"].astype(str)
    home_feats.index = home_feats.index.astype(str)
    away_feats.index = away_feats.index.astype(str)

    df = df.join(home_feats, on="game_id_str")
    df = df.join(away_feats, on="game_id_str")
    df = df.drop(columns=["game_id_str"])

    log.info(f"Merged auto features → shape={df.shape}")
    return df


# ── 主流程 ────────────────────────────────────────────────────

def build_auto_features(save: bool = True) -> pd.DataFrame:
    feat_path = FEAT_DIR / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found. Run manual_features.py first."
        )

    df = pd.read_csv(feat_path, parse_dates=["game_date"])
    log.info(f"Loaded manual features: {df.shape}")

    ts_df      = build_team_timeseries(df)
    auto_feats = extract_tsfresh_features(ts_df)
    auto_feats = select_features(auto_feats)
    df         = merge_auto_features(df, auto_feats)

    if save:
        out_path = FEAT_DIR / "features_full.csv"
        df.to_csv(out_path, index=False)
        log.info(f"Saved full features → {out_path}  shape={df.shape}")

    return df


if __name__ == "__main__":
    df = build_auto_features(save=True)
    print(f"\n最終特徵數：{len(df.columns)}  資料筆數：{len(df)}")

    # 顯示新增的 auto 特徵
    auto_cols = [c for c in df.columns if c.startswith(("h_auto_", "a_auto_"))]
    print(f"tsfresh 新增特徵數：{len(auto_cols)}")
    print("範例：", auto_cols[:10])
