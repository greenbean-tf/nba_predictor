"""
output.py
---------
把 Ensemble 的迴歸預測值轉換成：
  - 14 個 threshold 各自的 over/under
  - 機率（來自分類頭）
  - 信心等級
  - 建議下注清單
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


THRESHOLDS = [223.5, 224.5, 225.5, 226.5, 227.5, 228.5,
              229.5, 230.5, 231.5, 232.5, 233.5, 234.5, 235.5, 236.5]


def pred_to_signals(pred_total: float, pred_std: float = 10.0) -> pd.DataFrame:
    """
    把單一預測總分轉換成所有 threshold 的信號。

    Args:
        pred_total : 模型預測的總得分（e.g. 229.3）
        pred_std   : 預測的標準差（用來算機率，預設 10 分）

    Returns:
        DataFrame，每列是一個 threshold，欄位包含：
          threshold, direction, diff, prob_over, confidence, recommend
    """
    rows = []
    for line in THRESHOLDS:
        diff = pred_total - line
        direction = "OVER" if diff > 0 else "UNDER"

        # 機率：用常態分布估算（pred ~ N(pred_total, pred_std²)）
        prob_over = 1 - norm.cdf(line, loc=pred_total, scale=pred_std)
        prob = prob_over if direction == "OVER" else (1 - prob_over)

        # 信心等級（根據離線距離）
        abs_diff = abs(diff)
        if abs_diff >= 3.0:
            confidence = "高 ✅"
            recommend = True
        elif abs_diff >= 1.5:
            confidence = "中 ⚠️"
            recommend = False
        else:
            confidence = "低 ❌"
            recommend = False

        rows.append({
            "threshold": line,
            "direction": direction,
            "diff":      round(diff, 1),
            "prob":      round(prob, 3),
            "confidence": confidence,
            "recommend":  recommend,
        })

    return pd.DataFrame(rows)


def format_output(pred_total: float, pred_std: float = 10.0,
                  game_info: dict = None) -> str:
    """
    印出人類可讀的預測報告。

    Args:
        pred_total : 預測總分
        pred_std   : 預測標準差
        game_info  : {"home": "LAL", "away": "GSW", "quarter": "H1", "score": "63-55"}
    """
    df = pred_to_signals(pred_total, pred_std)

    lines = []
    lines.append("=" * 52)
    if game_info:
        lines.append(f"  {game_info.get('away','???')} @ {game_info.get('home','???')}")
        lines.append(f"  節次：{game_info.get('quarter','?')}  "
                     f"當前比分：{game_info.get('score','?')}")
    lines.append(f"  預測總得分：{pred_total:.1f}  (±{pred_std:.1f})")
    lines.append("=" * 52)
    lines.append(f"  {'線':>6}  {'方向':<6}  {'差距':>5}  {'機率':>6}  {'信心'}")
    lines.append("  " + "-" * 46)

    for _, row in df.iterrows():
        flag = " ◀" if row["recommend"] else ""
        lines.append(
            f"  {row['threshold']:>6.1f}  {row['direction']:<6}  "
            f"{row['diff']:>+5.1f}  {row['prob']:>5.1%}  "
            f"{row['confidence']}{flag}"
        )

    # 建議下注清單
    recs = df[df["recommend"]]
    lines.append("=" * 52)
    if recs.empty:
        lines.append("  本場無高信心建議（預測落在中間地帶）")
    else:
        lines.append("  建議下注：")
        for _, row in recs.iterrows():
            lines.append(
                f"    {row['threshold']} {row['direction']}  "
                f"(機率 {row['prob']:.1%})"
            )
    lines.append("=" * 52)

    return "\n".join(lines)


# ── 範例執行 ──────────────────────────────────────────────────
if __name__ == "__main__":
    example_info = {
        "home": "LAL",
        "away": "GSW",
        "quarter": "H1（半場）",
        "score": "63-55",
    }
    print(format_output(pred_total=229.3, pred_std=9.5, game_info=example_info))
