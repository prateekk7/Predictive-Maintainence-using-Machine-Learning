import streamlit as st
import pandas as pd
import numpy as np
import joblib


import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# TITLE
# -------------------------------
st.title("🔧 Predictive Maintenance System")

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("Enter Machine Parameters")

air_temp = st.sidebar.slider("Air Temperature (K)", 290, 320, 300)
process_temp = st.sidebar.slider("Process Temperature (K)", 290, 330, 310)
rpm = st.sidebar.slider("Rotational Speed (RPM)", 1000, 3000, 1500)
torque = st.sidebar.slider("Torque (Nm)", 10, 80, 40)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 300, 100)

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
Type_L = 1 if machine_type == "L" else 0
Type_M = 1 if machine_type == "M" else 0

power = torque * rpm
temp_diff = process_temp - air_temp

# -------------------------------
# INPUT DATAFRAME
# -------------------------------
input_data = pd.DataFrame([[ 
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear,
    Type_L,
    Type_M,
    temp_diff,
    power
]], columns=[
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Type_L',
    'Type_M',
    'temp_diff',
    'power'
])

input_scaled = scaler.transform(input_data)

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("Predict Machine Health"):

    risk_score = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    # -------------------------------
    # RESULT
    # -------------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Machine Failure Likely")
    else:
        st.success("✅ Machine Healthy")

    st.write(f"### Risk Score: {risk_score:.2f}")

    # -------------------------------
    # NEEDLE GAUGE
    # -------------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Failure Risk"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 0.3], 'color': "green"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # AI INSIGHTS
    # -------------------------------
    st.subheader("AI Insights")

    insight = ""

    if torque > 60:
        insight += "⚠️ High torque detected. "
    if tool_wear > 200:
        insight += "⚠️ Tool wear is critical. "
    if temp_diff > 15:
        insight += "⚠️ Overheating risk. "

    if insight == "":
        insight = "System operating under safe conditions."

    st.info(insight)

    # -------------------------------
    # CONFIDENCE BAR
    # -------------------------------
    st.subheader("Prediction Confidence")

    confidence = {
        "Safe": 1 - risk_score,
        "Failure": risk_score
    }

    st.bar_chart(confidence)

    # -------------------------------
    # INPUT FEATURE GRAPH
    # -------------------------------
    st.subheader("Input Feature Values")

    fig2, ax = plt.subplots()
    ax.barh(input_data.columns, input_data.values[0])
    st.pyplot(fig2)

    # -------------------------------
    # SENSITIVITY ANALYSIS
    # -------------------------------
    st.subheader("Sensitivity Analysis (Torque vs Risk)")

    torque_range = np.linspace(10, 80, 30)
    risk_values = []

    for t in torque_range:
        temp = input_data.copy()
        temp['Torque [Nm]'] = t
        temp['power'] = t * rpm

        temp_scaled = scaler.transform(temp)
        risk = model.predict_proba(temp_scaled)[0][1]
        risk_values.append(risk)

    fig3, ax = plt.subplots()
    ax.plot(torque_range, risk_values)
    ax.set_xlabel("Torque")
    ax.set_ylabel("Risk")

    st.pyplot(fig3)

    # -------------------------------
    # SIMULATION GRAPH
    # -------------------------------
    st.subheader("Risk Variation Simulation")

    sim_data = []

    for i in range(20):
        temp = input_data.copy()
        temp['Torque [Nm]'] += np.random.randint(-10, 10)
        temp['Tool wear [min]'] += np.random.randint(-20, 20)

        temp_scaled = scaler.transform(temp)
        prob = model.predict_proba(temp_scaled)[0][1]
        sim_data.append(prob)

    st.line_chart(sim_data)

# -------------------------------
# STATIC MODEL PERFORMANCE
# -------------------------------
st.subheader("📊 Model Performance")

try:
    df = pd.read_csv(r"C:\Major Project\pm_project\data_set.csv")
    df.columns = df.columns.str.strip()

    df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']

    df = pd.get_dummies(df, columns=['Type'], drop_first=True)

    y = df["Machine failure"]

    X = df[['Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Type_L',
            'Type_M',
            'temp_diff',
            'power']]

    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:,1]

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig4, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig4)

    # ROC Curve
    st.write("### ROC Curve")

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig5, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.legend()
    st.pyplot(fig5)

except Exception as e:
    st.error(f"Error: {e}")