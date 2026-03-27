import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from xgboost import XGBRegressor
import os

st.title("🏀 NBA Props – PRO System")

API_KEY = "ee00dc100b6b082fb9593ae7c187ec5a"

# -----------------------
# LOAD BET HISTORY
# -----------------------

if not os.path.exists("bets.csv"):
    pd.DataFrame(columns=["date","player","line","prediction","edge","pick","result","profit"]).to_csv("bets.csv", index=False)

bets_df = pd.read_csv("bets.csv")

# -----------------------
# GET ODDS
# -----------------------

def get_odds():
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={API_KEY}&markets=player_points"
    data = requests.get(url).json()
    
    # DEBUG: see what you're actually getting
    print("Response:", data)

    # If it's not a list, something went wrong
    if not isinstance(data, list):
        print("API Error:", data)
        return {}


    odds = {}

    for game in data:
            if not isinstance(game, dict):
            continue

        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    name = outcome["description"]
                    line = outcome.get("point")
                    if line:
                        odds[name] = line
    return odds

# -----------------------
# MODEL
# -----------------------

def get_prediction(player_name):
    try:
        nba_players = players.get_players()
        player_dict = [p for p in nba_players if p['full_name'] == player_name][0]
        player_id = player_dict['id']

        df = playergamelog.PlayerGameLog(player_id=player_id, season='2023-24').get_data_frames()[0]

        df = df[["GAME_DATE","PTS","MIN"]]
        df.columns = ["date","points","minutes"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df["last5_pts"] = df["points"].rolling(5).mean()
        df["last5_min"] = df["minutes"].rolling(5).mean()
        df["pts_std"] = df["points"].rolling(10).std()

        df = df.dropna()

        X = df[["last5_pts","last5_min","pts_std"]]
        y = df["points"]

        model = XGBRegressor(n_estimators=200)
        model.fit(X,y)

        latest = df.iloc[-1]

        pred = model.predict(pd.DataFrame([{
            "last5_pts": latest["last5_pts"],
            "last5_min": latest["last5_min"],
            "pts_std": latest["pts_std"]
        }]))[0]

        conf = max(0,10-latest["pts_std"])

        return pred, conf
    except:
        return None, None

# -----------------------
# FIND BETS
# -----------------------

if st.button("🔥 Find Bets"):
    odds = get_odds()
    results = []

    for player in odds:
        pred, conf = get_prediction(player)
        if pred:
            line = odds[player]
            edge = pred - line

            if abs(edge) > 2:
                pick = "OVER" if edge > 0 else "UNDER"

                results.append({
                    "Player": player,
                    "Line": line,
                    "Prediction": round(pred,2),
                    "Edge": round(edge,2),
                    "Pick": pick,
                    "Confidence": round(conf,2)
                })

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(by="Edge", ascending=False)
        st.dataframe(df.head(5))

        selected = st.selectbox("Select bet to track", df["Player"])

        if st.button("Add Bet"):
            bet = df[df["Player"] == selected].iloc[0]

            new_row = {
                "date": datetime.now(),
                "player": bet["Player"],
                "line": bet["Line"],
                "prediction": bet["Prediction"],
                "edge": bet["Edge"],
                "pick": bet["Pick"],
                "result": "",
                "profit": 0
            }

            bets_df.loc[len(bets_df)] = new_row
            bets_df.to_csv("bets.csv", index=False)

            st.success("Bet added!")

# -----------------------
# TRACK RESULTS
# -----------------------

st.subheader("📊 Bet History")

if not bets_df.empty:
    st.dataframe(bets_df)

    idx = st.number_input("Row to update", 0, len(bets_df)-1, 0)

    result = st.selectbox("Result", ["win","loss"])

    if st.button("Update Result"):
        if result == "win":
            bets_df.at[idx, "profit"] = 0.91
        else:
            bets_df.at[idx, "profit"] = -1

        bets_df.at[idx, "result"] = result
        bets_df.to_csv("bets.csv", index=False)

# -----------------------
# STATS
# -----------------------

st.subheader("💰 Performance")

if not bets_df.empty:
    total_profit = bets_df["profit"].sum()
    win_rate = (bets_df["result"]=="win").mean()

    st.write(f"Total Profit: {round(total_profit,2)} units")
    st.write(f"Win Rate: {round(win_rate*100,1)}%")