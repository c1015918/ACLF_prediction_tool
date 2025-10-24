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

# =========================
# =========================
# 数值型特征（含肌酐单位选择）
# =========================
numeric_features = ['HDL-C', 'Cr', 'PT', 'Globulin', 'Neutrophils']
default_values = {'HDL-C': 1.2, 'Cr': 70.0, 'PT': 12.0, 'Globulin': 30.0, 'Neutrophils': 4.0}

st.markdown(f"<h2 style='font-size:20px;'>{t['numeric_title']}</h2>", unsafe_allow_html=True)

# 初始化 session_state
for feature in numeric_features:
    key = f"input_{feature}"
    if key not in st.session_state:
        st.session_state[key] = default_values[feature]

if "cr_unit" not in st.session_state:
    st.session_state.cr_unit = "µmol/L"

# 用户输入区
for feature in numeric_features:
    label = t["feature_labels"][feature]

    if feature == "Cr":
        unit_label = "Creatinine unit:" if lang == "English" else "肌酐单位："
        unit_options = ["µmol/L", "mg/dL"]

        # 单位选择
        unit = st.radio(unit_label, unit_options, horizontal=True, key="unit_selector")

        # 根据单位调整显示默认值
        default_value = st.session_state[f"input_{feature}"] if unit == "µmol/L" else round(st.session_state[f"input_{feature}"]/88.4, 2)

        # 输入框
        input_label = f"{label.split('(')[0].strip()} ({unit})"
        creatinine_input = st.number_input(input_label, value=default_value, key=f"input_{feature}_display")

        # 转换成 µmol/L
        user_input[feature] = creatinine_input * 88.4 if unit == "mg/dL" else creatinine_input
    else:
        input_val = st.number_input(label, value=st.session_state[f"input_{feature}"], key=f"input_{feature}_display")
        user_input[feature] = input_val
# =========================
# 预测
# =========================
if st.button(t["predict_button"]):
    # 按训练模型顺序构造特征
    features = np.array([[user_input[f] for f in feature_names]], dtype=float)

    predicted_proba = model.predict_proba(features)[0]
    class1_prob = predicted_proba[1] * 100

    st.write(f"**{t['infection_prob']}：** {class1_prob:.1f}%")

    threshold = 0.365
    risk = t["high"] if class1_prob/100 >= threshold else t["low"]
    st.write(f"**{t['risk_result']}（{t['threshold']} {threshold:.3f}）：** {risk}")

    st.info(t["disclaimer"])
# =========================
# =========================
# 5️⃣ SHAP 可解释性可视化（点击按钮显示）
# =========================
# =========================
import sys
import os

# 计算 SHAP 值
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(pd.DataFrame([list(user_input.values())], columns=feature_names))
# 创建按钮
show_shap = st.button("Show SHAP Force Plot" if lang == "English" else "显示 SHAP 图")

if show_shap:
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

    # 避免特殊字符被解析
    original_values = original_values.astype(str)

    # 绘图
    plt.figure(figsize=(20, 18))  # 图形尺寸
    fig = shap.force_plot(
        base_value,
        shap_values_for_sample,
        original_values,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        text_rotation=10
    )

    # 获取当前 Axes
    ax = plt.gca()

    # 调整字体大小
    for label in ax.get_yticklabels():
        label.set_fontsize(17)
    for label in ax.get_xticklabels():
        label.set_fontsize(17)
    for text in ax.texts:
        text.set_fontsize(12)  # FX 标签字体

    # 调整布局
    plt.subplots_adjust(top=0.16, bottom=0.15, left=0.2, right=0.85)

    # 添加 base value 数值标注
    ax.axvline(base_value, color='gray', linestyle='--', linewidth=1)
    ax.text(
        base_value,
        ax.get_ylim()[1]*2.55,
        f'Base value: {base_value:.3f} ←',
        color='gray',
        fontsize=14,
        ha='center',
        va='top',
        backgroundcolor='white'
    )

    # 保存图像
    plt.tight_layout()
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    plt.close()

    # 恢复标准输出
    sys.stdout = sys.__stdout__

    # 在 Streamlit 展示图片
    st.image("shap_force_plot.png")
import streamlit as st
import shap
import numpy as np
import xgboost as xgb

# ========== 模拟训练模型并生成 SHAP 值（仅示例） ==========
# 你自己的模型可直接替换掉这部分
X = np.random.rand(10, 5)
y = np.random.randint(0, 2, 10)
model = xgb.XGBClassifier().fit(X, y)

# 取一个样本用于展示 SHAP 力图
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 取第一个样本示例
sample_idx = 0
base_value = explainer.expected_value
sample_features = X[sample_idx]
sample_shap_values = shap_values[sample_idx].values

# ✅ 正确生成可显示的 HTML 版本（避免“Visualization omitted”错误）
force_plot = shap.force_plot(
    base_value,
    sample_shap_values,
    sample_features,
    matplotlib=False
)
force_plot_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

# ✅ 使用顶部语言选择 lang 变量实现中英文切换
# ===============================================
# SHAP 力图详细解释（与主界面语言同步）
# ===============================================

if lang == "中文":
    with st.expander("🧩 点击查看 SHAP 力图详细解释"):
        st.markdown("""
        **SHAP 力图（SHAP Force Plot）** 用于解释单个样本的预测结果，展示了每个特征对模型输出的影响。

        **1️⃣ 基线值（Base Value）**  
        - 图中标记的 *base value*（如 -0.592）表示模型的平均输出，也就是在没有任何特征影响时模型的预测值。  
        - 它相当于模型在整个样本集上的“初始”预测水平。

        **2️⃣ 模型输出值（f(x)）**  
        - 图中显示的 *f(x)* 值是该样本的最终预测结果。  
        - 它等于基线值加上所有特征的 SHAP 值（贡献之和）：  
          `f(x) = base value + Σ(SHAP_i)`

        **3️⃣ 特征贡献（红色和蓝色箭头）**  
        - 🔴 **红色箭头**：表示该特征对预测结果有**正向贡献**（推高预测值）。  
        - 🔵 **蓝色箭头**：表示该特征对预测结果有**负向贡献**（降低预测值）。

        **4️⃣ 影响程度（箭头长度）**  
        - 箭头越长，说明该特征的 SHAP 值绝对值越大，对当前样本预测的影响越显著。  
        - 例如，一个长的红色箭头表示该特征强烈地推动模型做出高风险预测。

        **📘 总结**  
        - 左侧（蓝色）特征使模型预测值减小；  
        - 右侧（红色）特征使模型预测值增大；  
        - 中间的灰色垂直线表示模型的平均预测水平。
        """)
else:
    with st.expander("🧩 Click to view detailed SHAP Force Plot explanation"):
        st.markdown("""
        **SHAP Force Plot** is used to interpret the prediction of an individual sample by showing how each feature contributes to the model output.

        **1️⃣ Base Value**  
        - The *base value* (e.g., -0.592) represents the model’s average output when no features are applied.  
        - It can be considered the model’s “starting point” prediction before adding feature effects.

        **2️⃣ Model Output (f(x))**  
        - The *f(x)* indicates the final predicted value for this sample.  
        - It equals the base value plus the sum of all SHAP values (feature contributions):  
          `f(x) = base value + Σ(SHAP_i)`

        **3️⃣ Feature Contributions (Red and Blue Arrows)**  
        - 🔴 **Red arrows**: Features that push the prediction higher (positive contribution).  
        - 🔵 **Blue arrows**: Features that push the prediction lower (negative contribution).

        **4️⃣ Magnitude of Impact (Arrow Length)**  
        - Longer arrows represent features with larger absolute SHAP values, meaning stronger influence on the prediction.  
        - For example, a long red arrow indicates a feature strongly driving the model toward a higher-risk prediction.

        **📘 Summary**  
        - Features on the **left (blue)** decrease the predicted value;  
        - Features on the **right (red)** increase the predicted value;  
        - The **gray vertical line** represents the model’s average output.
        """)
