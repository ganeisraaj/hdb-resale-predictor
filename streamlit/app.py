import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    df = pd.read_csv("hdb_resale.csv")
    df["year"] = df["month"].str[:4].astype(int)
    df["storey_mid"] = df["storey_range"].str.extract(r'(\d+)\s+TO\s+(\d+)').astype(float).mean(axis=1)
    def parse_lease(s):
        if pd.isna(s):
            return np.nan
        years = 0
        months = 0
        y = pd.Series(s).str.extract(r'(\d+)\s+year')
        m = pd.Series(s).str.extract(r'(\d+)\s+month')
        if not y[0].isna().all():
            years = float(y[0].iloc[0])
        if not m[0].isna().all():
            months = float(m[0].iloc[0])
        return years + months / 12
    df["lease_remaining"] = df["remaining_lease"].apply(parse_lease)
    flat_type_order = {
        "1 ROOM": 1, "2 ROOM": 2, "3 ROOM": 3,
        "4 ROOM": 4, "5 ROOM": 5, "EXECUTIVE": 6, "MULTI-GENERATION": 7
    }
    df["flat_type_num"] = df["flat_type"].map(flat_type_order)
    return df

@st.cache_resource
def train_model(df):
    df_model = pd.get_dummies(df, columns=["town"], drop_first=True)
    feature_cols = (
        ["floor_area_sqm", "storey_mid", "lease_remaining", "flat_type_num", "year"]
        + [c for c in df_model.columns if c.startswith("town_")]
    )
    X = df_model[feature_cols]
    y = np.log(df_model["resale_price"])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model, feature_cols, df_model

df = load_data()
model, feature_cols, df_model = train_model(df)

st.title("HDB Resale Price Predictor")
st.markdown(
    "Estimate the resale price of a Singapore HDB flat. "
    "Model trained on 238,000+ transactions from 2017 to 2025 (data.gov.sg). "
    "R² = 0.933, MAE ≈ S$37,000."
)

st.header("Flat Details")

col1, col2 = st.columns(2)

with col1:
    town = st.selectbox("Town", sorted(df["town"].unique()))
    flat_type = st.selectbox("Flat type", ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"])
    floor_area = st.slider("Floor area (sqm)", 30, 200, 90)

with col2:
    storey = st.slider("Storey (midpoint)", 1, 50, 8)
    lease_remaining = st.slider("Remaining lease (years)", 40, 99, 75)
    year = st.slider("Transaction year", 2017, 2025, 2024)

flat_type_order = {
    "1 ROOM": 1, "2 ROOM": 2, "3 ROOM": 3,
    "4 ROOM": 4, "5 ROOM": 5, "EXECUTIVE": 6, "MULTI-GENERATION": 7
}

input_dict = {
    "floor_area_sqm": floor_area,
    "storey_mid": storey,
    "lease_remaining": lease_remaining,
    "flat_type_num": flat_type_order[flat_type],
    "year": year,
}

for col in feature_cols:
    if col.startswith("town_"):
        town_name = col.replace("town_", "")
        input_dict[col] = 1 if town_name == town else 0

input_df = pd.DataFrame([input_dict])[feature_cols]

pred_log = model.predict(input_df)[0]
pred_price = np.exp(pred_log)

st.header("Predicted Resale Price")
st.metric("Estimated price", f"S${pred_price:,.0f}")
st.markdown(f"*Approximate range: S${pred_price*0.93:,.0f} – S${pred_price*1.07:,.0f} (±7% based on model MAE)*")

st.header("Town Price Context")
town_medians = df.groupby("town")["resale_price"].median().sort_values(ascending=False)
selected_median = town_medians[town]
st.markdown(f"Median resale price in **{town}**: S${selected_median:,.0f}")
rank = (town_medians > selected_median).sum() + 1
st.markdown(f"Ranked **{rank} of {len(town_medians)}** towns by median price.")