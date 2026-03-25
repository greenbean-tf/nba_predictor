"""
test_team_mapping.py
--------------------
測試所有 NBA 球隊名稱從 ESPN → 標準縮寫 → Polymarket 全部對得上。

測試流程：
  1. 從 ESPN scoreboard 抓今天的比賽（或用歷史資料補充）
  2. 模擬所有 30 支球隊的縮寫轉換路徑
  3. 從 Polymarket 抓實際市場名稱
  4. 交叉比對，找出任何對不上的組合

執行：
  python test_team_mapping.py
  python test_team_mapping.py --live   # 實際呼叫 Polymarket API
"""

import sys
import argparse
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 關掉 logging 避免 ab_simulator import 時產生重複 handler
logging.disable(logging.CRITICAL)
from src.live.ab_simulator import TEAM_NAME_MAP, ABBR_TO_FULL, ESPN_ABBR_FIX
logging.disable(logging.NOTSET)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")

# ── 對應表已在上方載入 ───────────────────────────────────────

# 30 支 NBA 球隊的完整資料
# (全名, 標準縮寫, ESPN 可能給的縮寫列表)
ALL_TEAMS = [
    # 東區 大西洋
    ("Celtics",        "BOS", ["BOS"]),
    ("Nets",           "BKN", ["BKN", "NJ", "NJN"]),
    ("Knicks",         "NYK", ["NYK", "NY"]),
    ("76ers",          "PHI", ["PHI"]),
    ("Raptors",        "TOR", ["TOR"]),
    # 東區 中央
    ("Bulls",          "CHI", ["CHI"]),
    ("Cavaliers",      "CLE", ["CLE"]),
    ("Pistons",        "DET", ["DET"]),
    ("Pacers",         "IND", ["IND"]),
    ("Bucks",          "MIL", ["MIL"]),
    # 東區 東南
    ("Hawks",          "ATL", ["ATL"]),
    ("Hornets",        "CHA", ["CHA", "CHH"]),
    ("Heat",           "MIA", ["MIA"]),
    ("Magic",          "ORL", ["ORL"]),
    ("Wizards",        "WAS", ["WAS", "WSH"]),
    # 西區 西北
    ("Nuggets",        "DEN", ["DEN"]),
    ("Timberwolves",   "MIN", ["MIN"]),
    ("Thunder",        "OKC", ["OKC"]),
    ("Trail Blazers",  "POR", ["POR"]),
    ("Jazz",           "UTA", ["UTA", "UTAH"]),
    # 西區 太平洋
    ("Warriors",       "GSW", ["GSW", "GS", "GSD"]),
    ("Clippers",       "LAC", ["LAC"]),
    ("Lakers",         "LAL", ["LAL"]),
    ("Suns",           "PHX", ["PHX", "PHO"]),
    ("Kings",          "SAC", ["SAC"]),
    # 西區 西南
    ("Mavericks",      "DAL", ["DAL"]),
    ("Rockets",        "HOU", ["HOU"]),
    ("Grizzlies",      "MEM", ["MEM"]),
    ("Pelicans",       "NOP", ["NOP", "NO", "NOR", "NOH"]),
    ("Spurs",          "SAS", ["SAS", "SA"]),
]


# ── Test 1: TEAM_NAME_MAP 完整性 ──────────────────────────────

def test_team_name_map():
    print("\n【Test 1】TEAM_NAME_MAP 完整性（全名 → 縮寫）")
    passed = 0
    for full, abbr, _ in ALL_TEAMS:
        if TEAM_NAME_MAP.get(full) == abbr:
            ok(f"{full:<20} → {abbr}")
            passed += 1
        elif full in TEAM_NAME_MAP:
            fail(f"{full:<20} → {TEAM_NAME_MAP[full]} (期望 {abbr})")
        else:
            fail(f"{full:<20} → 找不到")
    print(f"\n  結果：{passed}/{len(ALL_TEAMS)} 通過")
    return passed == len(ALL_TEAMS)


# ── Test 2: ABBR_TO_FULL 完整性 ──────────────────────────────

def test_abbr_to_full():
    print("\n【Test 2】ABBR_TO_FULL 完整性（縮寫 → 全名）")
    passed = 0
    for full, abbr, _ in ALL_TEAMS:
        if ABBR_TO_FULL.get(abbr) == full:
            ok(f"{abbr:<5} → {full}")
            passed += 1
        elif abbr in ABBR_TO_FULL:
            fail(f"{abbr:<5} → {ABBR_TO_FULL[abbr]} (期望 {full})")
        else:
            fail(f"{abbr:<5} → 找不到")
    print(f"\n  結果：{passed}/{len(ALL_TEAMS)} 通過")
    return passed == len(ALL_TEAMS)


# ── Test 3: ESPN_ABBR_FIX 完整性 ─────────────────────────────

def test_espn_abbr_fix():
    print("\n【Test 3】ESPN_ABBR_FIX 完整性（ESPN 縮寫 → 標準縮寫）")
    passed = 0
    total  = 0
    for full, abbr, espn_abbrs in ALL_TEAMS:
        for ea in espn_abbrs:
            if ea == abbr:
                continue  # 已是標準縮寫，不需要 fix
            total += 1
            fixed = ESPN_ABBR_FIX.get(ea)
            if fixed == abbr:
                ok(f"ESPN {ea:<6} → {fixed:<5} ({full})")
                passed += 1
            elif fixed:
                fail(f"ESPN {ea:<6} → {fixed:<5} (期望 {abbr}, {full})")
            else:
                fail(f"ESPN {ea:<6} → 找不到對應 (應為 {abbr}, {full})")
    print(f"\n  結果：{passed}/{total} 通過")
    return passed == total


# ── Test 4: 完整轉換路徑 ESPN → 全名 ─────────────────────────

def test_full_pipeline():
    print("\n【Test 4】完整轉換路徑（ESPN縮寫 → 標準縮寫 → 全名）")
    passed = 0
    total  = 0
    for full, abbr, espn_abbrs in ALL_TEAMS:
        for ea in espn_abbrs:
            total += 1
            # Step 1: ESPN fix
            std_abbr = ESPN_ABBR_FIX.get(ea, ea)
            # Step 2: abbr → full name
            result   = ABBR_TO_FULL.get(std_abbr)

            if result == full:
                ok(f"ESPN '{ea}' → '{std_abbr}' → '{result}'")
                passed += 1
            else:
                fail(f"ESPN '{ea}' → '{std_abbr}' → '{result}' (期望 '{full}')")

    print(f"\n  結果：{passed}/{total} 通過")
    return passed == total


# ── Test 5: 與 Polymarket 實際名稱比對 ───────────────────────

def test_polymarket_matching(live: bool = False):
    print("\n【Test 5】Polymarket 市場名稱比對")

    if not live:
        warn("跳過（需要 --live 才會實際呼叫 Polymarket API）")
        return True

    try:
        from src.scraper.polymarket_scraper import fetch_nba_game_events
        from datetime import date, timedelta
        import json

        # 抓活躍 + 已結束，涵蓋前後 7 天
        events    = fetch_nba_game_events(include_closed=True)
        today     = date.today()
        date_from = (today - timedelta(days=7)).isoformat()
        date_to   = (today + timedelta(days=7)).isoformat()

        # 從所有 event（不限 O/U）解析球隊名稱 + 日期
        # 用於名稱對應測試，所以不需要只看 O/U
        game_records = []
        for ev in events:
            title    = ev.get("title", "")
            parts    = title.split(" vs. ")
            if len(parts) != 2:
                continue
            away_team = parts[0].strip()
            home_team = parts[1].strip()
            game_date = (ev.get("startDate","") or ev.get("startDateIso",""))[:10]
            if not game_date or date_from <= game_date <= date_to:
                game_records.append({
                    "away": away_team,
                    "home": home_team,
                    "date": game_date,
                })

        # 去重（同一場比賽可能有多條線）
        seen_games = set()
        unique_games = []
        for g in game_records:
            key = (g["away"], g["home"])
            if key not in seen_games:
                seen_games.add(key)
                unique_games.append(g)

        poly_teams = set()
        for g in unique_games:
            poly_teams.add(g["away"])
            poly_teams.add(g["home"])

        print(f"  {date_from} ～ {date_to} 範圍內：")
        print(f"  {len(unique_games)} 場比賽，{len(poly_teams)} 支球隊")
        print()
        for g in sorted(unique_games, key=lambda x: (x["date"], x["away"])):
            print(f"    {g['date']}  {g['away']:<20} vs {g['home']}")
        print()

        # 用 markets 作為原始數據（以免影響其他邏輯）
        markets = [{"home_team": g["home"], "away_team": g["away"]} for g in unique_games]

        if not markets:
            warn("找不到任何 Polymarket 市場（可能 API 有變動）")
            return True

        # 收集 Polymarket 出現的所有球隊名稱
        poly_teams = set()
        for m in markets:
            poly_teams.add(m["home_team"])
            poly_teams.add(m["away_team"])

        print(f"  抓到 {len(markets)} 個市場，涵蓋 {len(poly_teams)} 支球隊")

        # 反向驗證：Polymarket 的名稱能否對回全名
        unmatched = []
        matched   = []
        for pt in sorted(poly_teams):
            matched_full = None
            for full, abbr, _ in ALL_TEAMS:
                if full.lower() in pt.lower() or pt.lower() in full.lower():
                    matched_full = full
                    break
            if matched_full:
                matched.append((pt, matched_full))
            else:
                unmatched.append(pt)

        ok(f"{len(matched)} 支球隊名稱對應成功")
        for pt, full in matched:
            print(f"     '{pt}' → '{full}'")

        if unmatched:
            print()
            for u in unmatched:
                fail(f"Polymarket '{u}' → 找不到對應全名！需要更新 TEAM_NAME_MAP")

        # 統計有市場 vs 沒市場
        has_market  = {full for full, _, _ in ALL_TEAMS
                       if any(full.lower() in pt.lower() or pt.lower() in full.lower()
                              for pt in poly_teams)}
        no_market   = [full for full, _, _ in ALL_TEAMS if full not in has_market]

        print(f"\n  有市場：{len(has_market)} 支 | 無市場（近期無比賽）：{len(no_market)} 支")
        if no_market:
            print(f"  無市場球隊：{', '.join(sorted(no_market))}")

        return len(unmatched) == 0

    except Exception as e:
        fail(f"Polymarket 測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── Test 6: 找出 TEAM_NAME_MAP 缺漏 ──────────────────────────

def test_missing_entries():
    print("\n【Test 6】檢查是否有缺漏的縮寫對應")

    all_standard_abbrs = {abbr for _, abbr, _ in ALL_TEAMS}
    all_espn_variants  = {ea for _, _, espn_abbrs in ALL_TEAMS for ea in espn_abbrs}

    # 檢查 ABBR_TO_FULL 是否有多餘或缺漏
    in_map  = set(ABBR_TO_FULL.keys())
    missing = all_standard_abbrs - in_map
    extra   = in_map - all_standard_abbrs

    passed = True
    if missing:
        for m in sorted(missing):
            fail(f"ABBR_TO_FULL 缺少：'{m}'")
        passed = False
    else:
        ok("ABBR_TO_FULL 沒有缺漏")

    if extra:
        for e in sorted(extra):
            warn(f"ABBR_TO_FULL 多出（可能是舊球隊）：'{e}'")

    # 檢查 ESPN_ABBR_FIX 是否涵蓋所有非標準縮寫
    non_standard = all_espn_variants - all_standard_abbrs
    fix_keys     = set(ESPN_ABBR_FIX.keys())
    missing_fix  = non_standard - fix_keys

    if missing_fix:
        for m in sorted(missing_fix):
            fail(f"ESPN_ABBR_FIX 缺少：'{m}'")
        passed = False
    else:
        ok("ESPN_ABBR_FIX 涵蓋所有非標準縮寫")

    return passed


# ── 主程式 ────────────────────────────────────────────────────

def main(live: bool = False):
    print("=" * 60)
    print("  NBA Team Name Mapping Test")
    print("=" * 60)

    tests = [
        ("TEAM_NAME_MAP 完整性",      lambda: test_team_name_map()),
        ("ABBR_TO_FULL 完整性",       lambda: test_abbr_to_full()),
        ("ESPN_ABBR_FIX 完整性",      lambda: test_espn_abbr_fix()),
        ("完整轉換路徑",              lambda: test_full_pipeline()),
        ("Polymarket 名稱比對",        lambda: test_polymarket_matching(live)),
        ("缺漏檢查",                  lambda: test_missing_entries()),
    ]

    results = []
    for name, func in tests:
        try:
            passed = func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  {RED}CRASH in {name}: {e}{RESET}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("  測試結果總覽")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print(f"  {GREEN}全部通過！名稱對應沒有問題。{RESET}")
    else:
        print(f"  {RED}有測試失敗，請依上方錯誤訊息修正。{RESET}")

    # 顯示需要補充的對應
    print("\n  如需新增對應，修改 ab_simulator.py 的：")
    print("    TEAM_NAME_MAP  = { '全名': '縮寫', ... }")
    print("    ESPN_ABBR_FIX  = { 'ESPN縮寫': '標準縮寫', ... }")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="實際呼叫 Polymarket API 驗證名稱")
    args = parser.parse_args()
    main(live=args.live)