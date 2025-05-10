import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def simulate_data(num_records=300):  # Default to 300 records
    data = []
    start_date = pd.to_datetime("2023-01-01")
    for i in range(num_records):
        draw_date = start_date + pd.Timedelta(days=i * 3)
        main_numbers = sorted(random.sample(range(1, 70), 5))
        powerball = random.randint(1, 26)
        power_play = random.choice([2, 3, 4, 5, 10])
        data.append([draw_date.strftime("%Y-%m-%d")] + main_numbers + [powerball, power_play])
    columns = ["Draw_Date", "Num1", "Num2", "Num3", "Num4", "Num5", "Powerball", "Power_Play"]
    return pd.DataFrame(data, columns=columns)

def engineer_features(df):
    df["Total_Sum"] = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].sum(axis=1)
    df["High_Num_Count"] = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].apply(lambda row: sum(n > 35 for n in row), axis=1)
    df["Low_Num_Count"] = 5 - df["High_Num_Count"]
    df["Even_Count"] = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].apply(lambda row: sum(n % 2 == 0 for n in row), axis=1)
    df["Odd_Count"] = 5 - df["Even_Count"]
    df["Range_Span"] = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].max(axis=1) - df[["Num1", "Num2", "Num3", "Num4", "Num5"]].min(axis=1)
    prev_nums = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].shift(1)
    df["Repeated_Numbers"] = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].apply(
        lambda row: sum(row.isin(prev_nums.loc[row.name])), axis=1
    ).fillna(0)
    return df

def generate_unique_prediction(predicted_array, main_range=(1, 69), count=5):
    unique_numbers = set()
    for num in np.round(predicted_array).astype(int):
        clipped_num = np.clip(num, main_range[0], main_range[1])
        if clipped_num not in unique_numbers:
            unique_numbers.add(clipped_num)
        if len(unique_numbers) == count:
            break
    while len(unique_numbers) < count:
        unique_numbers.add(random.randint(main_range[0], main_range[1]))
    return np.sort(list(unique_numbers))

# --- Streamlit App Interface ---
st.title("🎰 Powerball Predictor App")
st.markdown("This app predicts the next possible Powerball numbers based on historical trends. *(Just for fun!)*")

# Always simulate 300 records
df = simulate_data()
df = engineer_features(df)

features = ["Total_Sum", "High_Num_Count", "Low_Num_Count", "Even_Count", "Odd_Count", "Range_Span", "Repeated_Numbers"]
target_columns = ["Num1", "Num2", "Num3", "Num4", "Num5", "Powerball"]

X = df[features]
y = df[target_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

regressor = MultiOutputRegressor(RandomForestRegressor(random_state=42))
regressor.fit(X_train, y_train)

if st.button("🎯 Generate Prediction"):
    latest_features = df[features].iloc[-1:].values
    latest_scaled = scaler.transform(latest_features)
    predicted_numbers = regressor.predict(latest_scaled)

    predicted_main = generate_unique_prediction(predicted_numbers[0][:5])
    predicted_powerball = int(np.clip(round(predicted_numbers[0][5]), 1, 26))

    st.subheader("🔮 Predicted Numbers")
    st.success(f"**Main Numbers:** {', '.join(map(str, predicted_main))}")
    st.success(f"**Powerball:** {predicted_powerball}")
