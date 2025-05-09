import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Data dummy
data = pd.DataFrame({
    'team_a_goals': [2, 1, 3, 0, 1, 2, 1],
    'team_b_goals': [1, 2, 2, 3, 0, 1, 1],
    'team_a_shots': [10, 8, 12, 6, 7, 9, 5],
    'team_b_shots': [6, 10, 8, 14, 6, 5, 4],
    'team_a_possession': [55, 60, 58, 50, 52, 56, 47],
    'team_b_possession': [45, 40, 42, 50, 48, 44, 53],
    'result': [1, 0, 1, 0, 1, 1, 0]  # 1 = Team A Menang, 0 = Team B Menang
})

# Training model
X = data.drop('result', axis=1)
y = data['result']
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("Prediksi Parlay Sepak Bola")

st.header("Masukkan Data Pertandingan")
match_data = []
num_matches = st.number_input("Jumlah pertandingan untuk parlay", min_value=1, max_value=10, value=3)

for i in range(num_matches):
    st.subheader(f"Pertandingan #{i+1}")
    team_a_goals = st.number_input(f"Team A Goals (Match {i+1})", key=f"ga_{i}")
    team_b_goals = st.number_input(f"Team B Goals (Match {i+1})", key=f"gb_{i}")
    team_a_shots = st.number_input(f"Team A Shots (Match {i+1})", key=f"sa_{i}")
    team_b_shots = st.number_input(f"Team B Shots (Match {i+1})", key=f"sb_{i}")
    team_a_possession = st.slider(f"Team A Possession (%) (Match {i+1})", 0, 100, 50, key=f"pa_{i}")
    team_b_possession = 100 - team_a_possession

    match_data.append([team_a_goals, team_b_goals, team_a_shots, team_b_shots, team_a_possession, team_b_possession])

if st.button("Prediksi & Tampilkan Parlay"):
    input_df = pd.DataFrame(match_data, columns=X.columns)
    predictions = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Hasil Prediksi")
    total_odds = 1.0
    for idx, result in enumerate(predictions):
        p_win = prediction_proba[idx][1]
        odds = round(1 / p_win, 2)
        total_odds *= odds
        st.markdown(f"**Pertandingan {idx+1}**: {'Team A Menang' if result == 1 else 'Team B Menang'} (Odds: {odds})")

    st.success(f"Total Odds Parlay: {round(total_odds, 2)}") 
