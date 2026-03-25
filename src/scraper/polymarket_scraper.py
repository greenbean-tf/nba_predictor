"""
polymarket_scraper.py - Polymarket NBA total markets scraper
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse, json, logging, re, time
from datetime import datetime, timezone
import requests, pandas as pd, yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RAW_DIR = Path(__file__).parent.parent.parent / CFG["data"]["raw_dir"]
RAW_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_PATH = RAW_DIR / "polymarket_snapshots.jsonl"

GAMMA_BASE   = "https://gamma-api.polymarket.com"
NBA_SERIES   = "10345"
NBA_GAME_TAG = "100639"


def fetch_nba_game_events(include_closed: bool = False):
    """
    抓取 Polymarket NBA 比賽市場。
    include_closed=True 時同時抓已結束的比賽（用於測試/回測）。
    """
    all_events = []
    seen_ids   = set()

    # 參數組合：先抓活躍的，再視需要抓已結束的
    param_sets = [
        {"active": "true", "closed": "false"},
    ]
    if include_closed:
        param_sets.append({"active": "false", "closed": "true"})

    for params in param_sets:
        offset = 0
        while True:
            try:
                r = requests.get(f"{GAMMA_BASE}/events",
                    params={"series_id": NBA_SERIES, "tag_id": NBA_GAME_TAG,
                            "order": "startTime", "ascending": "false",
                            "limit": 100, "offset": offset, **params},
                    timeout=15, proxies={})
                r.raise_for_status()
                batch = r.json()
            except Exception as e:
                log.error(f"Failed: {e}")
                break

            if not batch:
                break

            for ev in batch:
                if ev.get("id") not in seen_ids:
                    seen_ids.add(ev.get("id"))
                    all_events.append(ev)

            if len(batch) < 100:
                break  # 最後一頁
            offset += 100

    log.debug(f"fetch_nba_game_events: {len(all_events)} events total")
    return all_events


def parse_total_markets(events):
    results = []
    now = datetime.now(timezone.utc).isoformat()
    for event in events:
        slug  = event.get("slug", "")
        title = event.get("title", "")
        parts = title.split(" vs. ")
        away_team = parts[0].strip() if len(parts) == 2 else ""
        home_team = parts[1].strip() if len(parts) == 2 else ""
        for market in event.get("markets", []):
            q        = market.get("question", "")
            outcomes = market.get("outcomes", "[]")
            prices   = market.get("outcomePrices", "[]")
            # Only full-game totals
            if "O/U" not in q or "1H" in q:
                continue
            try:
                ol = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                pl = json.loads(prices)   if isinstance(prices, str)   else prices
            except Exception:
                continue
            if ol[:2] != ["Over", "Under"]:
                continue
            m = re.search(r"O/U\s+([\d.]+)", q)
            if not m:
                continue
            line = float(m.group(1))
            try:
                over_prob  = float(pl[0])
                under_prob = float(pl[1])
            except (IndexError, ValueError):
                continue
            # 過濾已結算市場（接近 0 或 1 代表已結算）
            if over_prob > 0.99 or over_prob < 0.01:
                continue
            results.append({
                "game_slug": slug, "home_team": home_team, "away_team": away_team,
                "question": q, "line": line, "over_prob": over_prob,
                "under_prob": under_prob, "condition_id": market.get("conditionId", ""),
                "fetched_at": now,
                "game_date": (event.get("startDate", "") or
                              event.get("startDateIso", "") or "")[:10],
            })
    return results


def save_snapshot(markets, trigger="manual"):
    with open(SNAPSHOT_PATH, "a", encoding="utf-8") as f:
        for m in markets:
            m["trigger"] = trigger
            f.write(json.dumps(m, default=str) + "\n")
    log.info(f"Saved {len(markets)} snapshots -> {SNAPSHOT_PATH.name}")


def load_snapshots():
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame()
    rows = []
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def print_today_odds(markets):
    if not markets:
        print("  No total markets found.")
        return
    from itertools import groupby
    ms = sorted(markets, key=lambda x: x["game_slug"])
    print(f"\n{'='*65}")
    print(f"{'  Polymarket NBA Totals':^65}")
    print(f"{'='*65}")
    print(f"  {'Match':<32}  {'Line':>6}  {'Over':>6}  {'Under':>6}")
    print(f"  {'-'*57}")
    for slug, group in groupby(ms, key=lambda x: x["game_slug"]):
        for m in group:
            label = f"{m['away_team']} @ {m['home_team']}"
            print(f"  {label:<32}  {m['line']:>6.1f}  {m['over_prob']:>5.1%}  {m['under_prob']:>5.1%}")
    print(f"{'='*65}\n")


def on_quarter_end(game, quarter):
    trigger = "Q1" if quarter == 1 else "halftime"
    log.info(f"[{trigger}] {game['away_team']} @ {game['home_team']}")
    events  = fetch_nba_game_events()
    markets = parse_total_markets(events)
    home = game["home_team"].lower()
    away = game["away_team"].lower()
    related = [m for m in markets
               if home in m["home_team"].lower() or away in m["away_team"].lower()
               or home in m["away_team"].lower() or away in m["home_team"].lower()]
    if related:
        save_snapshot(related, trigger=trigger)
        print_today_odds(related)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch",    action="store_true")
    parser.add_argument("--summary",  action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()

    if args.summary:
        df = load_snapshots()
        if df.empty:
            print("No snapshots yet.")
        else:
            print(f"Snapshots: {len(df)}, Games: {df['game_slug'].nunique()}")
        return

    def collect_once():
        events  = fetch_nba_game_events()
        markets = parse_total_markets(events)
        print_today_odds(markets)
        if markets:
            save_snapshot(markets, trigger="scheduled")
        return len(markets)

    if args.watch:
        log.info(f"Watch mode: every {args.interval}s")
        while True:
            collect_once()
            time.sleep(args.interval)
    else:
        collect_once()


if __name__ == "__main__":
    main()