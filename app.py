# --- CAREER TOTALS (Fixed Calculation) ---
try:
    career_df = playercareerstats.PlayerCareerStats(player_id=pid).get_data_frames()[0]
    if not career_df.empty:
        st.markdown("### Career Totals")

        # Sum across seasons for true totals
        totals = {
            "PTS": int(career_df["PTS"].sum()),
            "REB": int(career_df["REB"].sum()),
            "AST": int(career_df["AST"].sum()),
            "FG3M": int(career_df["FG3M"].sum()),
            "STL": int(career_df["STL"].sum()),
            "BLK": int(career_df["BLK"].sum()),
            "TOV": int(career_df["TOV"].sum())
        }

        # Format with commas
        totals_fmt = {k: f"{v:,}" for k, v in totals.items()}
        metric_cards(totals_fmt)

        # Bar chart for key totals
        bar_chart({k: totals[k] for k in ["PTS", "REB", "AST", "FG3M"] if k in totals},
                  "Career Totals — Key Stats")
except Exception as e:
    st.warning(f"Career stats unavailable: {e}")
