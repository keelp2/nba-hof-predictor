import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──
st.set_page_config(page_title="NBA Hall of Fame Predictor", page_icon="🏀", layout="wide")

# ── Load and prepare data ──
FEATURE_COLS = [
    "G", "PTS", "TRB", "AST", "STL", "BLK",
    "FG_PCT", "3P_PCT", "FT_PCT", "WS", "WS_PerMin",
    "TotalAllStar", "FirstAllD", "SecondAllD", "TotalAllD",
    "FirstAllNBA", "SecondAllNBA", "ThirdAllNBA", "TotalAllNBA",
]

FEATURE_LABELS = {
    "G": "Games", "PTS": "PPG", "TRB": "RPG", "AST": "APG",
    "STL": "SPG", "BLK": "BPG", "FG_PCT": "FG%", "3P_PCT": "3P%",
    "FT_PCT": "FT%", "WS": "Win Shares", "WS_PerMin": "WS/48",
    "TotalAllStar": "All-Star", "FirstAllD": "1st All-Def",
    "SecondAllD": "2nd All-Def", "TotalAllD": "All-Defense",
    "FirstAllNBA": "1st All-NBA", "SecondAllNBA": "2nd All-NBA",
    "ThirdAllNBA": "3rd All-NBA", "TotalAllNBA": "All-NBA",
}

def prob_color(p):
    if p >= 0.7: return "#2ecc71"
    if p >= 0.4: return "#f39c12"
    return "#e74c3c"

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df["HoF_label"] = (df["HoF"] == "Yes").astype(int)
    potentials = pd.read_csv("potentials.csv")
    return df, potentials

@st.cache_resource
def train_model(df):
    X = df[FEATURE_COLS].fillna(0)
    y = df["HoF_label"]

    model = RandomForestClassifier(n_estimators=500, random_state=12949, n_jobs=-1)
    model.fit(X, y)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=12949)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    importances = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    return model, cv_scores, importances

df, potentials = load_data()
model, cv_scores, importances = train_model(df)

# ── Predictions ──
df["HOF_Prob"] = model.predict_proba(df[FEATURE_COLS].fillna(0))[:, 1]
potentials["HOF_Prob"] = model.predict_proba(potentials[FEATURE_COLS].fillna(0))[:, 1]

all_players = pd.concat([
    df[["Name", "HoF", "HOF_Prob"] + FEATURE_COLS],
    potentials[["Name", "HoF", "HOF_Prob"] + FEATURE_COLS],
], ignore_index=True)

hof_yes = df[df["HoF"] == "Yes"]
hof_no = pd.concat([
    df[df["HoF"] == "No"],
    potentials,
], ignore_index=True).sort_values("HOF_Prob", ascending=False)

# HOF averages for radar chart
hof_avg = hof_yes[FEATURE_COLS].mean()

# ── Sidebar ──
st.sidebar.title("🏀 NBA HOF Predictor")
st.sidebar.caption(f"Trained on {len(df)} players · {len(potentials)} candidates")
st.sidebar.markdown(f"**Model accuracy:** {cv_scores.mean():.1%}")
st.sidebar.divider()

page = st.sidebar.radio("", ["🔮 Predict", "🏆 Top Candidates", "📊 Feature Importance", "🔄 Compare", "📋 Data", "ℹ️ About"])

# ══════════════════════════════════════════
# PREDICT PAGE
# ══════════════════════════════════════════
if page == "🔮 Predict":
    st.title("Player Prediction")

    player = st.selectbox("Select a candidate", potentials.sort_values("Name")["Name"].tolist())
    row = potentials[potentials["Name"] == player].iloc[0]
    prob = row["HOF_Prob"]
    actual = row["HoF"]
    color = prob_color(prob)

    # Probability donut ring
    col_gauge, col_stats = st.columns([1, 2])

    with col_gauge:
        fig = go.Figure(go.Pie(
            values=[prob * 100, 100 - prob * 100],
            hole=0.75,
            marker=dict(colors=[color, "#f0f0f0"], line=dict(width=0)),
            textinfo="none",
            hoverinfo="skip",
            sort=False,
        ))
        fig.add_annotation(
            text=f"<b>{prob:.0%}</b>",
            font=dict(size=36, color=color),
            showarrow=False, x=0.5, y=0.55,
        )
        fig.add_annotation(
            text="HOF Prob",
            font=dict(size=12, color="#999"),
            showarrow=False, x=0.5, y=0.38,
        )
        fig.update_layout(
            height=220, margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        status_icon = "✅" if actual == "Yes" else "⏳"
        st.markdown(f"<div style='text-align:center'><b>Status:</b> {status_icon} {'Inducted' if actual == 'Yes' else 'Not yet inducted'}</div>", unsafe_allow_html=True)

    with col_stats:
        st.markdown("##### Career Stats")
        stats_row1 = {
            "Games": f"{row['G']:.0f}", "PPG": f"{row['PTS']:.1f}",
            "RPG": f"{row['TRB']:.1f}", "APG": f"{row['AST']:.1f}",
            "FG%": f"{row['FG_PCT']:.3f}", "WS": f"{row['WS']:.1f}",
            "All-Star": f"{row['TotalAllStar']:.0f}",
        }
        stats_row2 = {
            "SPG": f"{row['STL']:.1f}", "BPG": f"{row['BLK']:.1f}",
            "3P%": f"{row['3P_PCT']:.3f}", "FT%": f"{row['FT_PCT']:.3f}",
            "WS/48": f"{row['WS_PerMin']:.3f}", "All-NBA": f"{row['TotalAllNBA']:.0f}",
            "All-Def": f"{row['TotalAllD']:.0f}",
        }
        st.dataframe(pd.DataFrame([stats_row1]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame([stats_row2]), hide_index=True, use_container_width=True)

    # Radar chart: player vs avg HOFer
    st.markdown("##### Player vs. Average HOF Member")
    radar_stats = ["PTS", "TRB", "AST", "STL", "BLK", "WS_PerMin"]
    radar_labels = [FEATURE_LABELS[s] for s in radar_stats]

    player_vals = [float(row[s]) for s in radar_stats]
    avg_vals = [float(hof_avg[s]) for s in radar_stats]

    # Scale each axis 0 to 1.1 * max(player, avg) so both traces fill the chart
    axis_max = [1.1 * max(p, a, 0.001) for p, a in zip(player_vals, avg_vals)]
    player_scaled = [100 * v / m for v, m in zip(player_vals, axis_max)]
    avg_scaled = [100 * v / m for v, m in zip(avg_vals, axis_max)]

    # Custom hover text showing actual values
    player_hover = [f"{label}: {val:.1f}" for label, val in zip(radar_labels, player_vals)]
    avg_hover = [f"{label}: {val:.1f}" for label, val in zip(radar_labels, avg_vals)]

    # Combined hover text: show both values at each point
    combined_hover = [
        f"<b>{l}</b><br>{player}: {pv:.1f}<br>Avg HOFer: {av:.1f}"
        for l, pv, av in zip(radar_labels, player_vals, avg_vals)
    ]
    combined_hover.append(combined_hover[0])  # close the loop

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_scaled + [avg_scaled[0]], theta=radar_labels + [radar_labels[0]],
        fill="toself", name="Avg HOFer", fillcolor="rgba(74,144,217,0.12)",
        line=dict(color="#4a90d9", width=2),
        mode="lines", hoverinfo="skip",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=player_scaled + [player_scaled[0]], theta=radar_labels + [radar_labels[0]],
        fill="toself", name=player, fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
        line=dict(color=color, width=2),
        mode="lines+markers", marker=dict(size=8, color=color),
        text=combined_hover, hoverinfo="text",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 105])),
        height=350, margin=dict(t=30, b=30, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Comparison histograms
    st.markdown("##### Distribution Among HOF Members")
    compare_stats = [
        ("TotalAllStar", "All-Star Selections"),
        ("WS", "Win Shares"),
        ("TotalAllNBA", "All-NBA Selections"),
        ("PTS", "Points Per Game"),
    ]
    cols = st.columns(2)
    for i, (col_name, label) in enumerate(compare_stats):
        with cols[i % 2]:
            fig = px.histogram(
                hof_yes, x=col_name, nbins=20,
                labels={col_name: label},
                color_discrete_sequence=["#4a90d9"],
                opacity=0.6,
            )
            fig.add_vline(
                x=row[col_name], line_dash="dash", line_color=color, line_width=2.5,
                annotation_text=player.split()[-1], annotation_position="top right",
                annotation_font_size=11,
            )
            fig.update_layout(
                height=250, margin=dict(t=25, b=25, l=30, r=20),
                showlegend=False, title_text=label, title_font_size=13,
                yaxis_title=None, xaxis_title=None,
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# TOP CANDIDATES PAGE
# ══════════════════════════════════════════
elif page == "🏆 Top Candidates":
    st.title("Top HOF Candidates")
    st.caption("Players not yet inducted, ranked by predicted probability.")

    for _, row in hof_no.head(15).iterrows():
        prob = row["HOF_Prob"]
        col1, col2, col3 = st.columns([3, 5, 2])
        with col1:
            st.markdown(f"**{row['Name']}**")
        with col2:
            st.progress(min(prob, 1.0))
        with col3:
            st.markdown(f"`{prob:.1%}`")

    with st.expander("View full table"):
        display = hof_no[["Name", "HOF_Prob", "G", "PTS", "TRB", "AST", "WS", "TotalAllStar", "TotalAllNBA"]].copy()
        display.columns = ["Player", "HOF Prob", "Games", "PPG", "RPG", "APG", "Win Shares", "All-Stars", "All-NBA"]
        display["HOF Prob"] = display["HOF Prob"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display.reset_index(drop=True), hide_index=True, use_container_width=True, height=400)

# ══════════════════════════════════════════
# FEATURE IMPORTANCE PAGE
# ══════════════════════════════════════════
elif page == "📊 Feature Importance":
    st.title("What Predicts a Hall of Famer?")
    st.caption("Feature importance from the 500-tree random forest model.")

    fig = px.bar(
        importances, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#e8e8e8", "#4a90d9", "#1a5276"],
    )
    fig.update_layout(
        height=520, margin=dict(t=10, b=30, l=10, r=30),
        coloraxis_showscale=False, yaxis_title=None, xaxis_title="Relative Importance",
        font=dict(size=13),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info("🏅 **Accolades dominate** — All-Star selections and All-NBA nods are the strongest signals.")
    with col2:
        st.info("📈 **Win Shares matter** — a holistic measure of a player's contribution carries more weight than any single counting stat.")

# ══════════════════════════════════════════
# COMPARE PAGE
# ══════════════════════════════════════════
elif page == "🔄 Compare":
    st.title("Compare Players")

    names = all_players.sort_values("Name")["Name"].tolist()
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player 1", names, index=names.index("Michael Jordan") if "Michael Jordan" in names else 0)
    with col2:
        default2 = names.index("Kobe Bryant") if "Kobe Bryant" in names else 1
        p2 = st.selectbox("Player 2", names, index=default2)

    r1 = all_players[all_players["Name"] == p1].iloc[0]
    r2 = all_players[all_players["Name"] == p2].iloc[0]

    # Side by side probability rings
    col1, col2 = st.columns(2)
    for col, r, name in [(col1, r1, p1), (col2, r2, p2)]:
        prob = r["HOF_Prob"]
        clr = prob_color(prob)
        with col:
            fig = go.Figure(go.Pie(
                values=[prob * 100, 100 - prob * 100],
                hole=0.78,
                marker=dict(colors=[clr, "#f0f0f0"], line=dict(width=0)),
                textinfo="none", hoverinfo="skip", sort=False,
            ))
            fig.add_annotation(
                text=f"<b>{prob:.0%}</b>",
                font=dict(size=28, color=clr),
                showarrow=False, x=0.5, y=0.55,
            )
            fig.add_annotation(
                text=name.split()[-1],
                font=dict(size=11, color="#999"),
                showarrow=False, x=0.5, y=0.38,
            )
            fig.update_layout(height=180, margin=dict(t=5, b=5, l=5, r=5), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    compare_fields = ["G", "PTS", "TRB", "AST", "STL", "BLK", "FG_PCT", "WS", "WS_PerMin", "TotalAllStar", "TotalAllNBA", "TotalAllD"]
    compare_df = pd.DataFrame({
        "Stat": [FEATURE_LABELS.get(f, f) for f in compare_fields],
        p1: [r1[f] for f in compare_fields],
        p2: [r2[f] for f in compare_fields],
    })
    st.dataframe(compare_df, hide_index=True, use_container_width=True)

    # Radar comparison
    radar_stats = ["PTS", "TRB", "AST", "STL", "BLK", "WS_PerMin"]
    radar_labels = [FEATURE_LABELS[s] for s in radar_stats]
    v1 = [float(r1[s]) for s in radar_stats]
    v2 = [float(r2[s]) for s in radar_stats]

    # Scale each axis 0 to 1.1 * max of both players
    axis_max = [1.1 * max(a, b, 0.001) for a, b in zip(v1, v2)]
    v1_scaled = [100 * v / m for v, m in zip(v1, axis_max)]
    v2_scaled = [100 * v / m for v, m in zip(v2, axis_max)]

    # Combined hover: show both values at each point
    combined_hover = [
        f"<b>{l}</b><br>{p1}: {a:.1f}<br>{p2}: {b:.1f}"
        for l, a, b in zip(radar_labels, v1, v2)
    ]
    combined_hover.append(combined_hover[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v1_scaled + [v1_scaled[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself", name=p1, fillcolor="rgba(74,144,217,0.12)",
        line=dict(color="#4a90d9", width=2),
        mode="lines", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=v2_scaled + [v2_scaled[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself", name=p2, fillcolor="rgba(231,76,60,0.12)",
        line=dict(color="#e74c3c", width=2),
        mode="lines+markers", marker=dict(size=8, color="#e74c3c"),
        text=combined_hover, hoverinfo="text",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 105])),
        height=400, margin=dict(t=30, b=40, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# DATA PAGE
# ══════════════════════════════════════════
elif page == "📋 Data":
    st.title("Full Dataset")
    st.caption(f"{len(all_players)} players ({len(df)} training + {len(potentials)} candidates)")

    show_hof = st.multiselect("Filter by HOF status", ["Yes", "No"], default=["Yes", "No"])
    filtered = all_players[all_players["HoF"].isin(show_hof)].sort_values("HOF_Prob", ascending=False)

    display = filtered[["Name", "HOF_Prob", "HoF"] + FEATURE_COLS].copy()
    display = display.rename(columns={"HOF_Prob": "HOF Prob", **FEATURE_LABELS})
    display["HOF Prob"] = display["HOF Prob"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display.reset_index(drop=True), hide_index=True, use_container_width=True, height=600)

# ══════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("About This Project")

    st.markdown("""
    A **random forest classifier** trained on career statistics and accolades
    of 323 NBA players to predict Hall of Fame induction probability.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Features (19):**
        - Career averages: PPG, RPG, APG, SPG, BPG
        - Shooting: FG%, 3P%, FT%
        - Impact: Games, Win Shares, WS/48
        - Accolades: All-Star, All-NBA, All-Defensive selections
        """)
    with col2:
        st.markdown(f"""
        **Model details:**
        - 500-tree random forest
        - 10-fold stratified cross-validation
        - Accuracy: **{cv_scores.mean():.1%}** ± {cv_scores.std():.1%}
        - Default `sqrt(n_features)` feature sampling
        """)

    st.divider()
    st.markdown("""
    **History:** Originally built in R/Shiny at Carleton College (2020). Rebuilt in Python/Streamlit (2026).

    **Data:** [Basketball Reference](https://www.basketball-reference.com/) · **Code:** [GitHub](https://github.com/keelp2) · **Site:** [keelp2.github.io](https://keelp2.github.io)
    """)
