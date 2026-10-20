import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="帕金森DBS神经动力学仿真可视化平台", layout="wide")

# ---------------------- 多个👀错开弹跳动画CSS ----------------------
bounce_html = """
<style>
@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-20px); }
}
.eye1 {
  font-size:44px;
  animation: bounce 0.6s ease-in-out infinite;
  display:inline-block;
}
.eye2 {
  font-size:44px;
  animation: bounce 0.6s ease-in-out 0.15s infinite;
  display:inline-block;
}
.eye3 {
  font-size:44px;
  animation: bounce 0.6s ease-in-out 0.3s infinite;
  display:inline-block;
}
</style>
<div style="text-align:center; margin-bottom:12px;">
  <span class="eye1">👀</span>&nbsp;&nbsp;
  <span class="eye2">👀</span>&nbsp;&nbsp;
  <span class="eye3">👀</span>
</div>
"""
st.markdown(bounce_html, unsafe_allow_html=True)

st.title("帕金森DBS神经动力学仿真可视化平台")
st.markdown("📌 两种使用模式：①内置仿真模型，可调DBS刺激参数；②上传自己导出的CSV仿真文件")

st.audio("像梦一样自由.mp3", loop=True, format="audio/mpeg")


@st.cache_data
def generate_dbs_data(freq, amp):
    t = np.linspace(0, 1000, 1000)
    omega = 2 * np.pi * freq / 1000
    stn = -55 + 12 * np.sin(omega * t) + amp * np.sin(2 * omega * t)
    gpe = -60 + 8 * np.sin(omega * t + 0.6) + amp * 0.7 * np.sin(2 * omega * t)
    gpi = -62 + 10 * np.sin(omega * t + 1.2) + amp * 0.8 * np.sin(2 * omega * t)
    df = pd.DataFrame({
        "time(ms)": t,
        "STN": stn,
        "GPe": gpe,
        "GPi": gpi
    })
    pulse_interval = 1000 / freq
    pulse_times = np.arange(pulse_interval / 2, 1000, pulse_interval)
    return df, pulse_times


mode = st.radio("选择数据来源", ["内置仿真模型（可调DBS参数，直接演示）", "上传自己的CSV文件"], horizontal=True)
df = None
pulse_times = []
title_text = "核团膜电位时序曲线"

if mode == "内置仿真模型（可调DBS参数，直接演示）":
    st.sidebar.header("⚙️ DBS刺激参数调节")
    stim_freq = st.sidebar.slider("刺激频率 (Hz)", min_value=5, max_value=60, value=13, step=1)
    stim_amp = st.sidebar.slider("刺激幅值 (mV)", min_value=0.0, max_value=8.0, value=3.0, step=0.5)
    df, pulse_times = generate_dbs_data(stim_freq, stim_amp)
    title_text = f"核团膜电位时序曲线 | DBS频率：{stim_freq} Hz，幅值：{stim_amp} mV"
    st.success(f"✅ 已生成DBS仿真 | 频率：{stim_freq} Hz，幅值：{stim_amp} mV")
else:
    upload_file = st.file_uploader("上传csv仿真数据文件", type=["csv"])
    if upload_file is not None:
        df = pd.read_csv(upload_file)
    else:
        st.info("⬆️ 请上传CSV格式仿真结果文件")


if df is not None:
    st.subheader("📋 数据预览")
    st.dataframe(df.head(15), use_container_width=True)

    time_col = "time(ms)"
    signal_cols = ["STN", "GPe", "GPi"]

    st.sidebar.header("📈 绘图设置")
    selected_cols = st.sidebar.multiselect(
        "选择需要绘制的核团信号",
        options=signal_cols,
        default=signal_cols
    )

    if selected_cols:
        fig = px.line(
            df,
            x=time_col,
            y=selected_cols,
            title=title_text,
            labels={"value": "膜电位 (mV)", time_col: "时间 (ms)", "variable": "核团"}
        )
        if mode == "内置仿真模型（可调DBS参数，直接演示）":
            for pt in pulse_times:
                fig.add_vline(x=pt, line_dash="dot", line_color="red", line_width=1.2)

        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 保存图片：点击图表右上角相机图标直接截图保存")
    else:
        st.warning("请至少勾选一个核团信号进行绘图")

# ========== 修改后的底部署名 ==========
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666666; font-size: 15px; line-height:1.6;'>"
    "豫北医学院·Powered by 王一浩 牛璐鸣 李雨轩<br>"
    "指导老师👩‍🏫：丰依凡（超厉害的！）"
    "</div>",
    unsafe_allow_html=True
)
