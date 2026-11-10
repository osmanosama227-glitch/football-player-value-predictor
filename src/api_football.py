"""Optional: pull a player's stats from API-Football to feed the app.
Key stays in an environment variable:  $env:API_FOOTBALL_KEY="..."
Free plan = 100 requests/day, so cache results and check which seasons your plan allows.
"""
import os
import requests

BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": os.getenv("API_FOOTBALL_KEY", "")}


def player_stats(name, league_id=39, season=2023):
    r = requests.get(f"{BASE}/players", headers=HEADERS, timeout=30,
                     params={"search": name, "league": league_id, "season": season})
    r.raise_for_status()
    out = []
    for item in r.json().get("response", []):
        s = item["statistics"][0]
        out.append({
            "name": item["player"]["name"], "age": item["player"]["age"],
            "position": s["games"]["position"],
            "minutes": s["games"]["minutes"] or 0,
            "apps": s["games"]["appearences"] or 0,   # (sic) API spelling
            "goals": s["goals"]["total"] or 0,
            "assists": s["goals"]["assists"] or 0,
        })
    return out


if __name__ == "__main__":
    print(player_stats("Saka"))
