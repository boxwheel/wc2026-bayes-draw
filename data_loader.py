"""
Wave 2 data loading and feature engineering for WC-2026 match prediction.
Pre-match features ONLY -- no leakage from match events, stats, or lineups.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "fifa_data"


def load_data():
    """Load and merge all pre-match features. Returns (X, y, match_ids)."""
    matches = pd.read_csv(DATA_DIR / "matches_detailed.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    squads = pd.read_csv(DATA_DIR / "squads_and_players.csv")
    venues = pd.read_csv(DATA_DIR / "venues.csv")

    # Keep only completed group-stage matches (64 labelled rows)
    completed = matches[matches["status"] == "Completed"].copy()
    assert len(completed) == 64, f"Expected 64 completed matches, got {len(completed)}"

    # Label: H / D / A
    def label(row):
        if row["home_score"] > row["away_score"]:
            return "H"
        elif row["home_score"] == row["away_score"]:
            return "D"
        else:
            return "A"

    completed["outcome"] = completed.apply(label, axis=1)

    # Squad aggregates per team (all pre-tournament)
    def squad_features(team_id, squads):
        ts = squads[squads["team_id"] == team_id].copy()
        if len(ts) == 0:
            return {}
        ts["market_value_eur"] = pd.to_numeric(ts["market_value_eur"], errors="coerce").fillna(0)
        ts["caps"] = pd.to_numeric(ts["caps"], errors="coerce").fillna(0)
        ts["goals"] = pd.to_numeric(ts["goals"], errors="coerce").fillna(0)
        ts["height_cm"] = pd.to_numeric(ts["height_cm"], errors="coerce").fillna(175)
        top11_mv = ts.nlargest(11, "market_value_eur")["market_value_eur"].sum()
        att_goals = ts[ts["position"].isin(["FW", "FWD", "ATT", "ST", "CF", "LW", "RW"])]["goals"].sum()
        gk = ts[ts["position"].isin(["GK"])]
        gk_mv = gk["market_value_eur"].max() if len(gk) > 0 else 0
        # Parse DOB for age (use 2026-06-01 as reference)
        try:
            ts["dob"] = pd.to_datetime(ts["date_of_birth"], errors="coerce")
            ref = pd.Timestamp("2026-06-01")
            ts["age"] = (ref - ts["dob"]).dt.days / 365.25
            mean_age = ts["age"].mean()
        except Exception:
            mean_age = 27.0
        return {
            "total_mv": ts["market_value_eur"].sum(),
            "top11_mv": top11_mv,
            "mean_caps": ts["caps"].mean(),
            "att_goals": att_goals,
            "gk_mv": gk_mv,
            "mean_height": ts["height_cm"].mean(),
            "mean_age": mean_age,
            "n_veterans": (ts["caps"] >= 50).sum(),
        }

    # Build team lookup
    teams_idx = teams.set_index("fifa_code")

    def get_team_row(fifa_code):
        return teams_idx.loc[fifa_code] if fifa_code in teams_idx.index else None

    # Host nations
    HOSTS = {"MEX", "USA", "CAN"}

    rows = []
    for _, match in completed.iterrows():
        h_code = match["home_fifa_code"]
        a_code = match["away_fifa_code"]
        h_team = get_team_row(h_code)
        a_team = get_team_row(a_code)

        def safe_get(row, col, default=0):
            try:
                return float(row[col]) if row is not None and col in row.index else default
            except Exception:
                return default

        h_elo = safe_get(h_team, "elo_rating", 1500)
        a_elo = safe_get(a_team, "elo_rating", 1500)
        h_rank = safe_get(h_team, "fifa_ranking_pre_tournament", 100)
        a_rank = safe_get(a_team, "fifa_ranking_pre_tournament", 100)
        h_conf = h_team["confederation"] if h_team is not None else "OTHER"
        a_conf = a_team["confederation"] if a_team is not None else "OTHER"

        h_tid = h_team["team_id"] if h_team is not None else -1
        a_tid = a_team["team_id"] if a_team is not None else -1
        h_sq = squad_features(h_tid, squads)
        a_sq = squad_features(a_tid, squads)

        # Venue features
        if "stadium_name" in match and not pd.isna(match.get("stadium_name", None)):
            stadium = match["stadium_name"]
            venue_row = venues[venues["stadium_name"] == stadium]
            if len(venue_row) > 0:
                capacity = float(venue_row.iloc[0].get("capacity", 60000))
                elevation = float(venue_row.iloc[0].get("elevation_meters", 0))
            else:
                capacity = 60000.0
                elevation = 0.0
        else:
            capacity = 60000.0
            elevation = 0.0

        row = {
            "match_id": match["match_id"],
            "outcome": match["outcome"],
            # Elo features
            "elo_diff": h_elo - a_elo,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "rank_diff": -(h_rank - a_rank),  # positive = home stronger
            "home_rank": h_rank,
            "away_rank": a_rank,
            # Host advantage
            "home_is_host": int(h_code in HOSTS),
            "away_is_host": int(a_code in HOSTS),
            "host_advantage": int(h_code in HOSTS) - int(a_code in HOSTS),
            # Confederation (categorical)
            "home_conf": h_conf,
            "away_conf": a_conf,
            # Squad features (differences)
            "mv_diff": h_sq.get("total_mv", 0) - a_sq.get("total_mv", 0),
            "top11_mv_diff": h_sq.get("top11_mv", 0) - a_sq.get("top11_mv", 0),
            "caps_diff": h_sq.get("mean_caps", 0) - a_sq.get("mean_caps", 0),
            "att_goals_diff": h_sq.get("att_goals", 0) - a_sq.get("att_goals", 0),
            "gk_mv_diff": h_sq.get("gk_mv", 0) - a_sq.get("gk_mv", 0),
            "height_diff": h_sq.get("mean_height", 0) - a_sq.get("mean_height", 0),
            "age_diff": h_sq.get("mean_age", 0) - a_sq.get("mean_age", 0),
            "veterans_diff": h_sq.get("n_veterans", 0) - a_sq.get("n_veterans", 0),
            # Absolute squad values
            "home_total_mv": h_sq.get("total_mv", 0),
            "away_total_mv": a_sq.get("total_mv", 0),
            # Venue
            "capacity": capacity,
            "elevation": elevation,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def encode_outcome(y_str):
    """Map H/D/A -> 0/1/2."""
    mapping = {"H": 0, "D": 1, "A": 2}
    return np.array([mapping[v] for v in y_str])


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df["outcome"].value_counts())
    print(df[["elo_diff", "rank_diff", "host_advantage", "mv_diff"]].describe())
