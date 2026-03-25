"""
test_orderbook.py
-----------------
檢查 OrderBookSummary 的 asks 排序方向
確認哪個 index 是最低 ask（best ask）

執行：
  python test_orderbook.py
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

client = ClobClient(
    host         = "https://clob.polymarket.com",
    key          = os.getenv("POLYMARKET_PRIVATE_KEY"),
    chain_id     = 137,
    signature_type = 1,
    funder       = os.getenv("POLYMARKET_FUNDER"),
    creds        = ApiCreds(
        api_key        = os.getenv("POLYMARKET_API_KEY"),
        api_secret     = os.getenv("POLYMARKET_API_SECRET"),
        api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE"),
    )
)

# Nuggets @ Suns 232.5
condition_id = "0x29bc0f500eb53efb8797a6fb63f7aef0be3c6e09beb7b6b371db24d4812ec632"
market = client.get_market(condition_id)
tokens = market.get("tokens", [])

for t in tokens:
    book = client.get_order_book(t["token_id"])
    asks = getattr(book, "asks", []) or []
    bids = getattr(book, "bids", []) or []

    print(f"\n{'='*50}")
    print(f"outcome = {t['outcome']}")
    print(f"token   = {t['token_id'][:20]}...")
    print(f"asks count = {len(asks)},  bids count = {len(bids)}")

    print("\n  asks（前5筆）：")
    for i, a in enumerate(asks[:5]):
        print(f"    [{i}] price={getattr(a,'price','?')}  size={getattr(a,'size','?')}")

    if len(asks) > 5:
        print(f"    ...")
        print(f"    [{len(asks)-1}] price={getattr(asks[-1],'price','?')}  size={getattr(asks[-1],'size','?')}")

    print("\n  bids（前5筆）：")
    for i, b in enumerate(bids[:5]):
        print(f"    [{i}] price={getattr(b,'price','?')}  size={getattr(b,'size','?')}")

    # 嘗試找 best ask（最低賣出價）
    ask_prices = [float(getattr(a, "price", 0)) for a in asks]
    if ask_prices:
        min_ask = min(ask_prices)
        max_ask = max(ask_prices)
        print(f"\n  ask 最低價：{min_ask}  最高價：{max_ask}")
        print(f"  asks[0].price = {ask_prices[0]}  ← {'最低' if ask_prices[0] == min_ask else '最高'}")
        print(f"  asks[-1].price = {ask_prices[-1]}  ← {'最低' if ask_prices[-1] == min_ask else '最高'}")

print("\n" + "="*50)
print("結論：best ask 應該用 asks[0] 還是 asks[-1]？")
print("      或是 min(asks, key=lambda x: x.price)?")