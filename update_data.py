"""
Scrape Basketball Reference to update all datasets with current data.
Sources (same as original R project):
  - HOF candidates: https://www.basketball-reference.com/friv/hof.fcgi
  - All-NBA: https://www.basketball-reference.com/awards/all_league_by_player.html
  - All-Defense: https://www.basketball-reference.com/awards/all_defense_by_player.html
  - All-Star: https://www.basketball-reference.com/awards/all_star_by_player.html
  - HOF inductees: https://www.basketball-reference.com/awards/hof.html
"""
import pandas as pd
import time

FINAL_COLS = [
    "Name", "G", "PTS", "TRB", "AST", "STL", "BLK", "FG_PCT", "3P_PCT", "FT_PCT",
    "WS", "WS_PerMin", "TotalAllStar", "FirstAllD", "SecondAllD", "TotalAllD",
    "FirstAllNBA", "SecondAllNBA", "ThirdAllNBA", "TotalAllNBA", "HoF",
]

def fetch(url):
    print(f"  Fetching {url}")
    tables = pd.read_html(url)
    return tables[0]

def flatten_cols(df):
    """Flatten multi-level column headers from Basketball Reference."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[1] if "Unnamed" in c[0] else c[1] for c in df.columns]
    # Drop repeated header rows
    first_col = df.columns[0]
    df = df[df[first_col] != first_col].reset_index(drop=True)
    return df

def to_numeric(df, exclude=("Player", "Name", "Category", "Lg")):
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def main():
    # ── 1. Candidates ──
    print("\n=== Candidates (non-HOF with 50+ WS) ===")
    candidates = flatten_cols(fetch("https://www.basketball-reference.com/friv/hof.fcgi"))
    candidates = to_numeric(candidates)
    print(f"  {len(candidates)} candidates")
    time.sleep(15)

    # ── 2. All-Star ──
    print("\n=== All-Star ===")
    all_star = flatten_cols(fetch("https://www.basketball-reference.com/awards/all_star_by_player.html"))
    all_star = to_numeric(all_star)
    all_star_clean = all_star[["Player", "Tot"]].rename(columns={"Tot": "TotalAllStar"})
    print(f"  {len(all_star_clean)} players")
    time.sleep(15)

    # ── 3. All-NBA ──
    print("\n=== All-NBA ===")
    all_nba = flatten_cols(fetch("https://www.basketball-reference.com/awards/all_league_by_player.html"))
    all_nba = to_numeric(all_nba)
    # Multi-level: columns are Tot, 1st, 2nd, 3rd, Tot (NBA), 1st, 2nd, Tot (ABA)
    # After flatten we get duplicate column names. Use positional.
    cols = list(all_nba.columns)
    # Player is col[1], Tot is col[2], NBA 1st=col[3], 2nd=col[4], 3rd=col[5], Tot=col[6], ABA 1st=col[7], 2nd=col[8], Tot=col[9]
    all_nba_clean = pd.DataFrame({
        "Player": all_nba.iloc[:, 1],
        "FirstAllNBA": all_nba.iloc[:, 3].fillna(0) + all_nba.iloc[:, 7].fillna(0),
        "SecondAllNBA": all_nba.iloc[:, 4].fillna(0) + all_nba.iloc[:, 8].fillna(0),
        "ThirdAllNBA": all_nba.iloc[:, 5].fillna(0),
        "TotalAllNBA": all_nba.iloc[:, 2].fillna(0),
    })
    print(f"  {len(all_nba_clean)} players")
    time.sleep(15)

    # ── 4. All-Defense ──
    print("\n=== All-Defense ===")
    all_def = flatten_cols(fetch("https://www.basketball-reference.com/awards/all_defense_by_player.html"))
    all_def = to_numeric(all_def)
    # Player=col[1], Tot=col[2], NBA 1st=col[3], 2nd=col[4], Tot=col[5], ABA 1st=col[6], 2nd=col[7], Tot=col[8]
    all_def_clean = pd.DataFrame({
        "Player": all_def.iloc[:, 1],
        "FirstAllD": all_def.iloc[:, 3].fillna(0) + all_def.iloc[:, 6].fillna(0),
        "SecondAllD": all_def.iloc[:, 4].fillna(0) + all_def.iloc[:, 7].fillna(0),
        "TotalAllD": all_def.iloc[:, 2].fillna(0),
    })
    print(f"  {len(all_def_clean)} players")
    time.sleep(15)

    # ── 5. HOF Inductees ──
    print("\n=== HOF Inductees ===")
    inductees_raw = flatten_cols(fetch("https://www.basketball-reference.com/awards/hof.html"))
    inductees_raw = to_numeric(inductees_raw)
    inductees_players = inductees_raw[inductees_raw["Category"] == "Player"].copy()
    inductees_players["Name"] = inductees_players["Name"].str.replace(r"^(\S+\s+\S+).*", r"\1", regex=True)
    # Rename to match
    ind = inductees_players.rename(columns={"FG%": "FG_PCT", "3P%": "3P_PCT", "FT%": "FT_PCT", "WS/48": "WS_PerMin"})
    ind = ind[["Name"] + [c for c in ["G", "PTS", "TRB", "AST", "STL", "BLK", "FG_PCT", "3P_PCT", "FT_PCT", "WS", "WS_PerMin"] if c in ind.columns]]
    ind["HoF"] = 1
    hof_names = set(ind["Name"].tolist())
    print(f"  {len(ind)} HOF players")

    # ── 6. Build training data + potentials ──
    print("\n=== Building datasets ===")

    cand = candidates.rename(columns={"Player": "Name", "To": "Year", "FG%": "FG_PCT", "3P%": "3P_PCT", "FT%": "FT_PCT", "WS/48": "WS_PerMin"})
    cand = cand[["Name", "Year"] + [c for c in ["G", "PTS", "TRB", "AST", "STL", "BLK", "FG_PCT", "3P_PCT", "FT_PCT", "WS", "WS_PerMin"] if c in cand.columns]]
    # Remove players already in HOF
    cand = cand[~cand["Name"].isin(hof_names)].copy()
    cand["HoF"] = 0

    # Split: retired by 2020 = training, after 2020 = potentials
    cutoff = 2020
    train_cand = cand[cand["Year"] <= cutoff].drop(columns=["Year"])
    pot_cand = cand[cand["Year"] > cutoff].copy()

    print(f"  HOF inductees: {len(ind)}")
    print(f"  Training candidates (retired <= {cutoff}): {len(train_cand)}")
    print(f"  Potential candidates (retired > {cutoff}): {len(pot_cand)}")

    def add_accolades(df):
        df = df.merge(all_star_clean, left_on="Name", right_on="Player", how="left").drop(columns=["Player"], errors="ignore")
        df = df.merge(all_def_clean, left_on="Name", right_on="Player", how="left").drop(columns=["Player"], errors="ignore")
        df = df.merge(all_nba_clean, left_on="Name", right_on="Player", how="left").drop(columns=["Player"], errors="ignore")
        df = df.fillna(0)
        return df

    # Training data
    data = pd.concat([ind, train_cand], ignore_index=True)
    stat_check = ["G", "PTS", "TRB", "AST", "STL", "BLK"]
    data = data[~data[stat_check].isna().all(axis=1)]
    data = add_accolades(data)
    data["HoF"] = data["HoF"].map({1: "Yes", 0: "No", 1.0: "Yes", 0.0: "No"})
    for c in FINAL_COLS:
        if c not in data.columns:
            data[c] = 0
    data = data[FINAL_COLS]
    data.to_csv("data.csv", index=False)
    print(f"\n  Saved data.csv: {len(data)} players ({(data['HoF']=='Yes').sum()} HOF, {(data['HoF']=='No').sum()} not)")

    # Potentials
    pot = pot_cand.drop(columns=["Year"], errors="ignore")
    pot = add_accolades(pot)
    pot["HoF"] = "No"
    for c in FINAL_COLS:
        if c not in pot.columns:
            pot[c] = 0
    pot = pot[FINAL_COLS]
    pot.to_csv("potentials.csv", index=False)
    print(f"  Saved potentials.csv: {len(pot)} candidates")

    # Quick verify
    print("\n=== Quick verify ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    feature_cols = FINAL_COLS[1:-1]
    X = data[feature_cols].fillna(0)
    y = (data["HoF"] == "Yes").astype(int)
    model = RandomForestClassifier(n_estimators=500, random_state=12949, n_jobs=-1)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=12949)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"  10-fold CV accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    model.fit(X, y)
    X_pot = pot[feature_cols].fillna(0)
    pot_probs = model.predict_proba(X_pot)[:, 1]
    pot_display = pot[["Name"]].copy()
    pot_display["prob"] = pot_probs
    print(f"\n  Top potential candidates:")
    for _, r in pot_display.nlargest(10, "prob").iterrows():
        print(f"    {r['Name']}: {r['prob']:.1%}")

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
