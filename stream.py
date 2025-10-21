import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap  # <- 一定要加
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
st.markdown("<h1 style='text-align: center; font-size: 30px;'>Prediction Tool for Nosocomial Infections in ACLF</h1>", unsafe_allow_html=True)

# =========================
# 3️⃣ 用户输入界面
# =========================
user_input = {}

# 二分类特征（是/否）
binary_features = ['Antibiotics', 'Cerebral Failure',  'Circulatory Failure', 'HE']
st.markdown("<h2 style='font-size: 20px;'>Binary Feature (Yes/No)</h2>", unsafe_allow_html=True)
for feature in binary_features:
    choice = st.selectbox(f"{feature}:", ["No", "Yes"], index=0)
    user_input[feature] = 1 if choice == "Yes" else 0

# 数值型特征
numeric_features = ['HDL-C', 'Cr', 'PT', 'Globulin', 'Neutrophils']
st.markdown("<h2 style='font-size: 20px;'>Numerical Feature</h2>", unsafe_allow_html=True)
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
# =========================
# 5️⃣ SHAP 可解释性可视化（改进版，显示 base value）
# =========================
import sys
import os

# 计算 SHAP 值
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(pd.DataFrame([list(user_input.values())], columns=feature_names))

# 二分类任务选择正类
if isinstance(sv, list):
    shap_values = sv[1]  # 正类
else:
    shap_values = sv

# 获取基准值
if isinstance(explainer.expected_value, list):
    base_value = explainer.expected_value[1]
else:
    base_value = explainer.expected_value

# 重定向标准输出屏蔽多余信息（可选）
sys.stdout = open(os.devnull, 'w')

# 当前样本 SHAP 值和特征原始值
shap_values_for_sample = shap_values[0]
original_values = pd.DataFrame([list(user_input.values())], columns=feature_names).iloc[0]

# 绘图
plt.figure(figsize=(20, 18))  # 图形尺寸
fig = shap.force_plot(
    base_value,
    shap_values_for_sample,
    original_values,
    feature_names=feature_names,
    matplotlib=True,
    show=False,
    text_rotation=0
)

# 获取当前 Axes
ax = plt.gca()

# 调整字体大小
for label in ax.get_yticklabels():
    label.set_fontsize(17)
for label in ax.get_xticklabels():
    label.set_fontsize(17)
# 调整 FX 标签字体大小
for text in ax.texts:
    text.set_fontsize(16)  # FX 标签字体
# 调整布局：把图例往上移，避免遮住坐标轴
plt.subplots_adjust(top=0.19, bottom=0.15, left=0.2, right=0.85)

# 添加 base value 数值标注
ax.axvline(base_value, color='gray', linestyle='--', linewidth=1)
ax.text(base_value, ax.get_ylim()[1]*2.55, f'Base value: {base_value:.3f}', 
        color='gray', fontsize=14, ha='center', va='top', backgroundcolor='white')

# 保存图像
plt.tight_layout()
plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
plt.close()  # 避免内存泄漏

# 恢复标准输出
sys.stdout = sys.__stdout__

# 在 Streamlit 展示图片
st.image("shap_force_plot.png")
