import streamlit as st
import joblib
import pandas as pd
import json
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Delivery Risk Predictor",
    page_icon="📦",
    layout="wide",
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size:60px;
    font-weight:bold;
    color:#4CAF50;
}
.sub-text {
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model + Columns
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_columns():
    with open("columns.json", "r") as f:
        return json.load(f)

model = load_model()
data_columns = load_columns()

seller_states = data_columns["seller_states"]
customer_states = data_columns["customer_states"]
categories = data_columns["categories"]

# -----------------------------
# Title Section
# -----------------------------
st.markdown('<p class="main-title">📦 E-Commerce Delivery Risk Predictor</p>', unsafe_allow_html=True)

st.markdown("""
<p class="sub-text">
Predicts probability of delivery delay using operational order features.
Helps logistics teams take proactive decisions.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📝 Order Details")

payment_type = st.sidebar.selectbox(
    "Payment Type",
    ["credit_card", "cash", "voucher", "debit_card"]
)

seller_state = st.sidebar.selectbox("Seller State", seller_states)
customer_state = st.sidebar.selectbox("Customer State", customer_states)
product_category = st.sidebar.selectbox("Product Category", categories)
price = st.sidebar.number_input("Product Price", min_value=0.0)

# -----------------------------
# Input DataFrame
# -----------------------------
input_df = pd.DataFrame([{
    "payment_type": payment_type,
    "seller_state": seller_state,
    "customer_state": customer_state,
    "product_category_name_english": product_category,
    "price": price
}])

# Show Input Summary
st.subheader("📋 Order Summary")
st.dataframe(input_df, use_container_width=True)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔮 Predict Delivery Risk"):

    with st.spinner("Analyzing delivery risk..."):
        time.sleep(1)  # Visual effect
        proba = model.predict_proba(input_df)[0]
        classes = model.classes_

    prob_dict = dict(zip(classes, proba))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    predicted_label = sorted_probs[0][0]
    predicted_prob = sorted_probs[0][1]

    late_prob = prob_dict.get("Late", 0)

    st.markdown("---")

    # -----------------------------
    # Metrics Section
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        if predicted_label == "Late":
            st.error(f"🚨 Predicted: Late Delivery ({predicted_prob:.0%})")
        else:
            st.success(f"✅ Predicted: On-Time Delivery ({predicted_prob:.0%})")

    with col2:
        st.metric("Late Delivery Risk", f"{late_prob:.0%}")

    # -----------------------------
    # Risk Messages
    # -----------------------------
    if late_prob > 0.6:
        st.warning("⚠ High risk of delay detected!")
    elif late_prob > 0.4:
        st.info("⚠ Moderate risk of delay.")
    else:
        st.success("🎉 Delivery likely on time.")

    # -----------------------------
    # Probability Chart
    # -----------------------------
    st.subheader("📊 Prediction Probabilities")

    prob_df = pd.DataFrame(sorted_probs, columns=["Status", "Probability"])
    st.bar_chart(prob_df.set_index("Status"))

    # -----------------------------
    # Explanation Section
    # -----------------------------
    st.subheader("🧠 Model Explanation")

    st.write("""
    This ML model predicts delivery delays using:
    
    • Payment Type  
    • Seller Location  
    • Customer Location  
    • Product Category  
    • Product Price  
    
    Higher delay probability suggests increased logistics complexity.
    """)
