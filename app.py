import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bank Customer Churn Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* ---------------------------------------------------
MAIN APPLICATION THEME
--------------------------------------------------- */

.stApp {
    background-color: #0B132B;
}

/* ---------------------------------------------------
SIDEBAR
--------------------------------------------------- */

section[data-testid="stSidebar"] {
    background-color: #081C2D;
}

/* ---------------------------------------------------
KPI CARDS
--------------------------------------------------- */

div[data-testid="metric-container"] {

    background: linear-gradient(
        135deg,
        #1F4E79,
        #0A2540
    );

    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow:
        0px 4px 20px rgba(0,0,0,0.35);
    text-align:center;
}

/* ---------------------------------------------------
KPI LABELS
--------------------------------------------------- */

div[data-testid="metric-container"] label {

    color:white !important;
    font-size:14px !important;
    font-weight:600;
}

/* ---------------------------------------------------
KPI VALUES
--------------------------------------------------- */

div[data-testid="metric-container"] div {

    color:white !important;
}

/* ---------------------------------------------------
PAGE TITLE
--------------------------------------------------- */

.dashboard-title {

    background-color: #1C2541;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

/* ---------------------------------------------------
TABS
--------------------------------------------------- */

button[data-baseweb="tab"] {

    font-size:16px;
    font-weight:600;
}

/* ---------------------------------------------------
HEADINGS
--------------------------------------------------- */

h1,h2,h3 {

    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD TRAINED ARTIFACTS
# ---------------------------------------------------

model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---------------------------------------------------
# DASHBOARD HEADER
# ---------------------------------------------------

st.markdown("""
<div class="dashboard-title">
<h1>Bank Customer Churn Analytics Dashboard</h1>
<h4>Predictive Modeling and Risk Scoring System</h4>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# CUSTOMER INPUT PANEL
# ---------------------------------------------------

st.sidebar.header("Customer Information")

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

age = st.sidebar.slider(
    "Age",
    18,
    90,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    0.0,
    300000.0,
    60000.0
)

products = st.sidebar.slider(
    "Products",
    1,
    4,
    2
)

active = st.sidebar.toggle(
        "Active Member",
        value=True
    )
    
active = int(active)

credit_card = st.sidebar.toggle(
    "Credit Card",
    value=True
)

credit_card = int(credit_card)

salary = st.sidebar.number_input(
    "Salary",
    1000.0,
    300000.0,
    50000.0
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

country = st.sidebar.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)

balance_salary_ratio = balance/(salary+1)

age_tenure = age*tenure

engagement_product = products*active

input_df = pd.DataFrame(
    columns=feature_columns
)

input_df.loc[0] = 0

input_df["CreditScore"] = credit_score
input_df["Age"] = age
input_df["Tenure"] = tenure
input_df["Balance"] = balance
input_df["NumOfProducts"] = products
input_df["HasCrCard"] = credit_card
input_df["IsActiveMember"] = active
input_df["EstimatedSalary"] = salary

input_df["BalanceSalaryRatio"] = balance_salary_ratio
input_df["AgeTenure"] = age_tenure
input_df["EngagementProduct"] = engagement_product

if "Gender_Male" in input_df.columns:
    input_df["Gender_Male"] = (
        1 if gender=="Male" else 0
    )
    
if "Geography_Germany" in input_df.columns:
    input_df["Geography_Germany"] = (
        1 if country=="Germany" else 0
    )

if "Geography_Spain" in input_df.columns:
    input_df["Geography_Spain"] = (
        1 if country=="Spain" else 0
    )

# ---------------------------------------------------
# MODEL PREDICTION
# ---------------------------------------------------
    
probability = model.predict_proba(
    input_df
)[0][1]

# ---------------------------------------------------
# RISK SCORING
# ---------------------------------------------------

risk_score = round(
    probability*100,
    2
)

if risk_score >= 70:
    risk = "High Risk"

elif risk_score >= 40:
    risk = "Medium Risk"

else:
    risk = "Low Risk"
    
st.title("Predictive Modeling and Risk Scoring for Bank Customer Churn")

# ---------------------------------------------------
# KPI DASHBOARD
# ---------------------------------------------------

c1,c2,c3 = st.columns(3)

c1.metric(
    "Churn Probability",
    f"{risk_score}%"
)

c2.metric(
    "Risk Category",
    risk
)

c3.metric(
    "Retention Probability",
    f"{100-risk_score}%"
)

# ---------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------

tab1,tab2,tab3,tab4 = st.tabs(
    [
        "Risk Assessment",
        "Probability Distribution",
        "Churn Drivers",
        "Scenario Analysis"
    ]
)

with tab1:
    
    st.subheader("Customer Churn Risk")

    if risk_score < 40:
        bar_color = "#2E8B57"      # Green

    elif risk_score < 70:
        bar_color = "#D4A017"      # Gold

    else:
        bar_color = "#B22222"      # Red
        
    col1, col2 = st.columns([1,4])

    with col1:
        st.metric(
            "Risk Category",
            risk
        )

    with col2:
        st.markdown(f"""
        <div style="margin-top:10px;">
            <div style="
                width:100%;
                background-color:#D9E2EC;
                border-radius:20px;
                height:45px;
                overflow:hidden;
            ">
                <div style="
                    width:{risk_score}%;
                    background-color:{bar_color};
                    height:45px;
                    border-radius:20px;
                    text-align:center;
                    line-height:45px;
                    color:white;
                    font-size:18px;
                    font-weight:bold;
                ">
                    {risk_score:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    

with tab2:

    probs = np.random.normal(
        risk_score,
        10,
        500
    )

    fig = px.histogram(
        probs,
        nbins=25,
        color_discrete_sequence=["#006D77"],
        
    )
    
    fig.update_layout(
        showlegend=False,
        title="Predicted Churn Probability Distribution",
        xaxis_title="Churn Probability (%)",
        yaxis_title="Number of Customers",
        margin=dict(
            l=40,
            r=100,
            t=50,
            b=40
        )
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
imp = pd.read_csv(
    "feature_importance.xls"
)

with tab3:

    fig = px.bar(
        imp.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#1F4E79"]
    )

    fig.update_layout(
        title="Top Factors Influencing Customer Churn",
        xaxis_title="Importance Score",
        margin=dict(
            l=60,
            r=20,
            t=50,
            b=40
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
with tab4:

    st.subheader(
        "What-if Analysis"
    )
    
    scenario = input_df.copy()

    scenario["NumOfProducts"] = products

    scenario["IsActiveMember"] = active

    scenario["EngagementProduct"] = (
        products*active
    )

    scenario_prob = model.predict_proba(
        scenario
    )[0][1]

    scenario_score = round(
        scenario_prob*100,
        2
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Current",
            x=["Current"],
            y=[risk_score],
            marker_color="#1F4E79"
        )
    )

    fig.add_trace(
        go.Bar(
            name="Scenario",
            x=["Scenario"],
            y=[scenario_score],
            marker_color="#006D77"
        )
    )
    
    fig.update_layout(
        title="What-If Scenario Analysis",
        xaxis_title="Customer State",
        yaxis_title="Churn Probability (%)",
        margin=dict(
            l=40,
            r=100,
            t=50,
            b=40
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )