import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys, os

plt.rcParams['text.usetex'] = False  # 禁用 LaTeX mathtext

# =========================
# 语言选择
# =========================
lang = st.sidebar.selectbox("Language / 语言", ["English", "中文"])

# 中英文字典
text = {
    "English": {
        "title": "Prediction Tool for Nosocomial Infections in ACLF",
        "binary_title": "Binary Features (Yes/No)",
        "numeric_title": "Numerical Features",
        "predict_button": "Predict",
        "infection_prob": "Probability of Infection",
        "risk_result": "Risk Assessment",
        "high": "High Risk",
        "low": "Low Risk",
        "threshold": "Threshold",
        "disclaimer": "Disclaimer: This result is for reference only and should not be used for diagnosis or treatment decisions.",
        "feature_labels": {
            "Antibiotics": "Antibiotics",
            "Cerebral Failure": "Cerebral Failure",
            "Circulatory Failure": "Circulatory Failure",
            "HE": "Hepatic Encephalopathy",
            "HDL-C": "HDL-C (mmol/L)",
            "Cr": "Creatinine (µmol/L)",
            "PT": "Prothrombin Time (s)",
            "Globulin": "Globulin (g/L)",
            "Neutrophils": "Neutrophils (×10⁹/L)"
        }
    },
    "中文": {
        "title": "ACLF院内感染风险预测工具",
        "binary_title": "二分类特征（是/否）",
        "numeric_title": "数值型特征",
        "predict_button": "预测",
        "infection_prob": "院内感染概率",
        "risk_result": "风险评估",
        "high": "高风险",
        "low": "低风险",
        "threshold": "阈值",
        "disclaimer": "免责声明：本结果仅供参考，不可作为诊断或治疗依据。",
        "feature_labels": {
            "Antibiotics": "使用抗生素",
            "Cerebral Failure": "脑功能衰竭",
            "Circulatory Failure": "循环衰竭",
            "HE": "肝性脑病",
            "HDL-C": "高密度脂蛋白胆固醇 (mmol/L)",
            "Cr": "肌酐 (µmol/L)",
            "PT": "凝血酶原时间 (秒)",
            "Globulin": "球蛋白 (g/L)",
            "Neutrophils": "中性粒细胞 (×10⁹/L)"
        }
    }
}

# 当前语言文本
t = text[lang]

# =========================
# 1️⃣ 加载模型
# =========================
MODEL_PATH = "XGBmodel.pkl"
model = joblib.load(MODEL_PATH)

# =========================
# 2️⃣ 定义特征
# =========================
feature_names = [
    'Antibiotics', 'Cerebral Failure', 'Circulatory Failure', 'HE',
    'HDL-C', 'Cr', 'PT', 'Globulin', 'Neutrophils'
]

# =========================
# 页面标题
# =========================
st.markdown(f"<h1 style='text-align: center; font-size: 30px;'>{t['title']}</h1>", unsafe_allow_html=True)

# =========================
# 3️⃣ 用户输入界面
# =========================
user_input = {}

# 二分类特征
binary_features = ['Antibiotics', 'Cerebral Failure', 'Circulatory Failure', 'HE']
st.markdown(f"<h2 style='font-size: 20px;'>{t['binary_title']}</h2>", unsafe_allow_html=True)
for feature in binary_features:
    label = t["feature_labels"][feature]
    choice = st.selectbox(f"{label}:", ["No", "Yes"] if lang == "English" else ["否", "是"], index=0)
    user_input[feature] = 1 if (choice in ["Yes", "是"]) else 0

# 数值型特征
numeric_features = ['HDL-C', 'Cr', 'PT', 'Globulin', 'Neutrophils']
st.markdown(f"<h2 style='font-size: 20px;'>{t['numeric_title']}</h2>", unsafe_allow_html=True)
default_values = {
    'HDL-C': 1.2,
    'Cr': 70.0,
    'PT': 12.0,
    'Globulin': 30.0,
    'Neutrophils': 4.0
}
for feature in numeric_features:
    label = t["feature_labels"][feature]
    user_input[feature] = st.number_input(f"{label}:", value=default_values[feature])

# =========================
# 4️⃣ 预测
# =========================
if st.button(t["predict_button"]):
    features = np.array([list(user_input.values())])
    predicted_proba = model.predict_proba(features)[0]
    class1_prob = predicted_proba[1] * 100

    st.write(f"**{t['infection_prob']}：** {class1_prob:.1f}%")

    threshold = 0.365
    risk = t["high"] if class1_prob / 100 >= threshold else t["low"]
    st.write(f"**{t['risk_result']}（{t['threshold']} {threshold:.3f}）：** {risk}")

    st.info(t["disclaimer"])
