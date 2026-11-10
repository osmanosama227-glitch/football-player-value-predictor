from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
for p in (RAW, PROCESSED, MODELS):
    p.mkdir(parents=True, exist_ok=True)

DATASET_FILE = PROCESSED / "dataset.csv"
MODEL_FILE = MODELS / "best_model.joblib"
METRICS_FILE = MODELS / "metrics.csv"
LOOKUP_FILE = PROCESSED / "players_lookup.csv"
RANGE_FILE = MODELS / "error_by_range.csv"
HORIZON_DAYS = 365   # forecast horizon: value ~12 months after the reference date

LEAGUE_NAMES = {
    "GB1": "Premier League (England)", "GB2": "Championship (England)",
    "ES1": "La Liga (Spain)", "ES2": "La Liga 2 (Spain)",
    "IT1": "Serie A (Italy)", "IT2": "Serie B (Italy)",
    "L1": "Bundesliga (Germany)", "L2": "2. Bundesliga (Germany)",
    "FR1": "Ligue 1 (France)", "FR2": "Ligue 2 (France)",
    "NL1": "Eredivisie (Netherlands)", "PO1": "Liga Portugal",
    "TR1": "Super Lig (Turkey)", "BE1": "Pro League (Belgium)",
    "SC1": "Scottish Premiership", "RU1": "Premier League (Russia)",
    "UKR1": "Premier League (Ukraine)", "GR1": "Super League (Greece)",
    "DK1": "Superliga (Denmark)", "MLS1": "MLS (USA)",
    "BRA1": "Serie A (Brazil)", "ARG1": "Primera Division (Argentina)",
    "SA1": "Saudi Pro League", "unknown": "Unknown / other",
}


def league_name(code):
    return LEAGUE_NAMES.get(code, code)  # unmapped codes show as-is
