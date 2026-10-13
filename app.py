import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cloud IDS", layout="wide")
st.title("Cloud Intrusion Detection System")
st.caption("ML-based network traffic classifier (Random Forest, trained on NSL-KDD)")

@st.cache_resource
def load_bundle():
    return joblib.load("ids_model.joblib")

bundle = load_bundle()
model, scaler, columns = bundle["model"], bundle["scaler"], bundle["columns"]
classes = list(model.classes_)
normal_idx = classes.index("Normal")

st.sidebar.header("Detection settings")
threshold = st.sidebar.slider(
    "Alert threshold", 0.05, 0.90, 0.30, 0.05,
    help="Lower = catches more attacks but raises more false alarms.")
st.sidebar.caption("Lower threshold = higher detection rate, more false alarms.")

def analyze(raw, threshold):
    raw = raw.drop(columns=[c for c in ["label", "difficulty", "target", "category"]
                            if c in raw.columns])
    X = pd.get_dummies(raw, columns=["protocol_type", "service", "flag"])
    X = X.reindex(columns=columns, fill_value=0).astype(float)
    proba = model.predict_proba(scaler.transform(X))
    attack_prob = 1 - proba[:, normal_idx]
    masked = proba.copy()
    masked[:, normal_idx] = -1
    attack_type = np.array(classes)[masked.argmax(axis=1)]
    out = raw.copy()
    out["attack_probability"] = attack_prob.round(3)
    out["prediction"] = np.where(attack_prob >= threshold, attack_type, "Normal")
    return out

uploaded = st.file_uploader("Upload network traffic (CSV with header)", type="csv")
use_sample = st.checkbox("Use sample traffic instead")

data = None
if uploaded is not None:
    data = pd.read_csv(uploaded)
elif use_sample:
    data = pd.read_csv("sample_traffic.csv")

if data is not None:
    result = analyze(data, threshold)
    alerts = result[result["prediction"] != "Normal"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Flows analysed", len(result))
    c2.metric("Alerts raised", len(alerts))
    c3.metric("Alert rate", f"{len(alerts) / len(result):.1%}")

    if len(alerts):
        st.error(f"{len(alerts)} suspicious flows detected")
    else:
        st.success("No intrusions detected")

    st.subheader("Traffic breakdown")
    st.bar_chart(result["prediction"].value_counts())

    st.subheader("Alerts")
    show_cols = ["prediction", "attack_probability", "protocol_type",
                 "service", "src_bytes", "dst_bytes"]
    st.dataframe(alerts[show_cols], use_container_width=True)
    st.download_button("Download alert log", alerts.to_csv(index=False),
                       "alerts.csv", "text/csv")
else:
    st.info("Upload a CSV or tick 'Use sample traffic' to start.")