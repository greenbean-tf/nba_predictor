import requests

proxies = {}

# NBA series_id = 10345, tag_id = 100639（單場比賽）
r = requests.get(
    "https://gamma-api.polymarket.com/events",
    params={
        "series_id": "10345",
        "tag_id":    "100639",
        "active":    "true",
        "closed":    "false",
        "order":     "startTime",
        "ascending": "true",
        "limit":     20,
    },
    proxies=proxies
)
events = r.json()
print(f"Found {len(events)} NBA game events")

for e in events:
    print(f"\nEvent: {e.get('slug')}")
    print(f"Title: {e.get('title')}")
    for m in e.get("markets", []):
        q = m.get("question", "")
        prices = m.get("outcomePrices", "")
        outcomes = m.get("outcomes", "")
        print(f"  Q: {q}")
        print(f"  outcomes: {outcomes}")
        print(f"  prices:   {prices}")