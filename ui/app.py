"""Streamlit UI: ввод признаков вина → запрос к API → показ quality."""
import os

import requests
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")
st.title("🍷 Предсказание качества белого вина")
st.caption("Модель: RandomForestRegressor · API → сохранённая модель из models/")

api_url = os.environ.get("API_URL", "http://localhost:8000")

st.subheader("Физико-химические характеристики")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed acidity (г/дм³)", min_value=0.0, value=7.0, step=0.1)
    volatile_acidity = st.number_input("Volatile acidity (г/дм³)", min_value=0.0, value=0.27, step=0.01)
    citric_acid = st.number_input("Citric acid (г/дм³)", min_value=0.0, value=0.36, step=0.01)
    residual_sugar = st.number_input("Residual sugar (г/дм³)", min_value=0.0, value=20.7, step=0.1)
    chlorides = st.number_input("Chlorides (г/дм³)", min_value=0.0, value=0.045, step=0.001, format="%.3f")
    free_sulfur_dioxide = st.number_input("Free sulfur dioxide (мг/дм³)", min_value=0.0, value=45.0, step=1.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total sulfur dioxide (мг/дм³)", min_value=0.0, value=170.0, step=1.0)
    density = st.number_input("Density (г/см³)", min_value=0.9, value=1.001, step=0.0001, format="%.4f")
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=3.0, step=0.01)
    sulphates = st.number_input("Sulphates (г/дм³)", min_value=0.0, value=0.45, step=0.01)
    alcohol = st.number_input("Alcohol (% об.)", min_value=0.0, value=8.8, step=0.1)

payload = {
    "fixed_acidity": fixed_acidity,
    "volatile_acidity": volatile_acidity,
    "citric_acid": citric_acid,
    "residual_sugar": residual_sugar,
    "chlorides": chlorides,
    "free_sulfur_dioxide": free_sulfur_dioxide,
    "total_sulfur_dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
}

if st.button("Предсказать quality", type="primary"):
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        st.success(f"**Predicted quality: {data['predicted_quality']}**")
        st.write(f"Интерпретация: {data.get('interpretation', '')}")
        st.caption(data.get("unit", ""))
    except requests.RequestException as error:
        st.error(f"API недоступен или вернул ошибку: {error}")
