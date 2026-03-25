"""
test_espn.py
------------
驗證 ESPN API 能正確抓到：
  1. 比賽前（pre）狀態
  2. 比賽中（in）狀態
  3. 比賽結束（post）最終比分
  4. 各節比分

執行：
  python test_espn.py
"""

import requests
import json
from datetime import datetime

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"


def fetch():
    r = requests.get(URL, timeout=10, proxies={})
    r.raise_for_status()
    return r.json()


def main():
    print(f"\n抓取時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    data   = fetch()
    events = data.get("events", [])
    # 去除重複的比賽（ESPN 有時會回傳重複）
    seen = set()
    unique_events = []
    for e in events:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique_events.append(e)
    events = unique_events
    print(f"今日共 {len(events)} 場比賽\n")

    for ev in events:
        comp   = ev["competitions"][0]
        status = ev["status"]["type"]
        state  = status["state"]          # pre / in / post
        detail = status.get("detail", "")

        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")

        home_abbr = home["team"]["abbreviation"]
        away_abbr = away["team"]["abbreviation"]
        home_score = home.get("score", "-")
        away_score = away.get("score", "-")

        print(f"{'─'*55}")
        print(f"  {away_abbr} @ {home_abbr}")
        print(f"  狀態：{state}  ({detail})")
        print(f"  比分：{away_score} - {home_score}")

        # 各節比分
        line_scores = comp.get("competitors", [])
        for competitor in line_scores:
            side   = competitor["homeAway"]
            abbr   = competitor["team"]["abbreviation"]
            linesc = competitor.get("linescores", [])
            if linesc:
                quarters = [ls.get("value", "-") for ls in linesc]
                print(f"  {side:4} {abbr}: {quarters}")

        # 如果是結束狀態，顯示最終總分
        if state == "post":
            try:
                total = int(home.get("score", 0)) + int(away.get("score", 0))
                print(f"  ✅ 最終總分：{total}")
            except Exception:
                print(f"  ⚠️  無法計算總分")

        # 如果是進行中，顯示當前節次
        if state == "in":
            period = ev["status"].get("period", 0)
            clock  = ev["status"].get("displayClock", "")
            print(f"  🏀 第 {period} 節  {clock}")

        # 如果是賽前，顯示開賽時間
        if state == "pre":
            start = ev.get("date", "")
            print(f"  ⏰ 開賽時間：{start}")

    print(f"\n{'─'*55}")
    print("驗證重點：")
    print("  pre  = 比賽尚未開始（賽前模型觸發時機）")
    print("  in   = 比賽進行中（live_scraper 追蹤各節）")
    print("  post = 比賽結束（自動結算觸發時機）")
    print(f"{'─'*55}\n")


def check_abbr_mapping():
    """驗證 ESPN 縮寫能正確對應到球隊全名。"""
    ESPN_ABBR_FIX = {
        "GS": "GSW", "NO": "NOP", "SA": "SAS", "NY": "NYK",
        "UTAH": "UTA", "PHO": "PHX",
    }
    ABBR_TO_FULL = {
        "CLE": "Cavaliers", "CHI": "Bulls",   "DET": "Pistons", "WAS": "Wizards",
        "LAC": "Clippers",  "NOP": "Pelicans","LAL": "Lakers",  "MIA": "Heat",
        "MIL": "Bucks",     "UTA": "Jazz",    "ORL": "Magic",   "CHA": "Hornets",
        "PHI": "76ers",     "SAC": "Kings",   "PHX": "Suns",    "SAS": "Spurs",
        "GSW": "Warriors",  "BKN": "Nets",    "BOS": "Celtics", "MEM": "Grizzlies",
        "ATL": "Hawks",     "DAL": "Mavericks","DEN": "Nuggets","OKC": "Thunder",
        "TOR": "Raptors",   "MIN": "Timberwolves","POR": "Trail Blazers",
        "HOU": "Rockets",   "NYK": "Knicks",  "IND": "Pacers",
        "WAS": "Wizards",
    }

    print("\n縮寫對應驗證：")
    data   = fetch()
    events = data.get("events", [])
    all_ok = True
    for ev in events:
        comp = ev["competitions"][0]
        for c in comp["competitors"]:
            raw   = c["team"]["abbreviation"]
            fixed = ESPN_ABBR_FIX.get(raw, raw)
            full  = ABBR_TO_FULL.get(fixed, None)
            ok    = "✅" if full else "❌ 找不到對應"
            if not full:
                all_ok = False
            print(f"  ESPN={raw:6} → 修正={fixed:5} → {ok} {full or ''}")

    if all_ok:
        print("\n✅ 所有縮寫都能正確對應！")
    else:
        print("\n⚠️  有縮寫無法對應，需要更新 ESPN_ABBR_FIX")


if __name__ == "__main__":
    main()
    check_abbr_mapping()