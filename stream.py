import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
# =========================
# 1️⃣ 加载模型
# =========================
MODEL_PATH = "XGBmodel.pkl"
model = joblib.load(MODEL_PATH)
# =========================
# 2️⃣ 定义特征（顺序必须和训练一致）
# =========================
feature_names = [
    'Antibiotics', 'Cerebral Failure', 'Circulatory Failure', 'HE',
    'HDL-C', 'Cr', 'PT', 'Globulin', 'Neutrophils'
]

# Streamlit 页面标题
st.title("Prediction Tool for Nosocomial Infections in ACLF")

# =========================
# 3️⃣ 用户输入界面
# =========================
user_input = {}

# 二分类特征（是/否）
binary_features = ['Antibiotics', 'Cerebral Failure',  'Circulatory Failure', 'HE']
st.subheader("Binary Feature (Yes/No) ")
for feature in binary_features:
    choice = st.selectbox(f"{feature}:", ["No", "Yes"], index=0)
    user_input[feature] = 1 if choice == "Yes" else 0

# 数值型特征
numeric_features = ['HDL-C', 'Cr', 'PT', 'Globulin', 'Neutrophils']
st.subheader("Numerical feature")
default_values = {
    'HDL-C': 1.2,        # mmol/L
    'Cr': 70.0,          # µmol/L
    'PT': 12.0,          # 秒
    'Globulin': 30.0,    # g/L
    'Neutrophils': 4.0   # x10^9/L
}
for feature in numeric_features:
    user_input[feature] = st.number_input(f"{feature}:", value=default_values[feature])

# =========================
# 4️⃣ 预测按钮
# =========================
if st.button("Predict"):
    # 构建特征数组
    features = np.array([list(user_input.values())])

    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 明确类别对应概率
    class0_prob = predicted_proba[0] * 100  # 假设类别0 = 低风险
    class1_prob = predicted_proba[1] * 100  # 类别1 = 高风险

    # 显示概率
    st.write(f"**Probability of Infection:** {class1_prob:.1f}%")

    # 风险判定（阈值）
    threshold = 0.365
    risk = "High Risk" if class1_prob / 100 >= threshold else "Low Risk"
    st.write(f"**Risk Assessment (Threshold {threshold:.3f}):** {risk}")
    # =========================
    # 5️⃣ SHAP 可解释性可视化
    # =========================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([list(user_input.values())], columns=feature_names))

    # 绘制 force plot 并保存为图片
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        pd.DataFrame([list(user_input.values())], columns=feature_names), 
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption="SHAP Force Plot: Feature Contribution")
