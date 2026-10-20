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

# 音频：像梦一样自由.mp3，上传到仓库根目录
st.audio("像梦一样自由.mp3", loop=True, format="audio/mpeg")


@st.cache_data
def generate_dbs_data(freq, amp, pulse_width, stim_strength, total_time, sample_num):
    t = np.linspace(0, total_time, sample_num)
    omega = 2 * np.pi * freq / 1000
    # 基础膜电位叠加DBS刺激脉冲调制
    pulse_mod = np.where(np.mod(t, 1000/freq) < pulse_width, stim_strength, 0)
    stn = -55 + 12 * np.sin(omega * t) + amp * np.sin(2 * omega * t) + pulse_mod*0.4
    gpe = -60 + 8 * np.sin(omega * t + 0.6) + amp * 0.7 * np.sin(2 * omega * t) + pulse_mod*0.25
    gpi = -62 + 10 * np.sin(omega * t + 1.2) + amp * 0.8 * np.sin(2 * omega * t) + pulse_mod*0.35
    df = pd.DataFrame({
        "time(ms)": t,
        "STN": stn,
        "GPe": gpe,
        "GPi": gpi
    })
    pulse_interval = 1000 / freq
    pulse_times = np.arange(pulse_interval / 2, total_time, pulse_interval)
    return df, pulse_times


mode = st.radio("选择数据来源", ["内置仿真模型（可调DBS参数，直接演示）", "上传自己的CSV文件"], horizontal=True)
df = None
pulse_times = []
title_text = "核团膜电位时序曲线"

if mode == "内置仿真模型（可调DBS参数，直接演示）":
    st.sidebar.header("⚙️ DBS刺激参数调节")
    stim_freq = st.sidebar.slider("刺激频率 (Hz)", min_value=5, max_value=60, value=13, step=1)
    stim_amp = st.sidebar.slider("调制幅值 (mV)", min_value=0.0, max_value=8.0, value=3.0, step=0.5)
    pulse_width = st.sidebar.slider("刺激脉宽 (ms)", min_value=0.1, max_value=2.0, value=0.4, step=0.1)
    stim_strength = st.sidebar.slider("刺激强度", min_value=0.0, max_value=5.0, value=2.0, step=0.2)
    total_time = st.sidebar.slider("仿真总时长 (ms)", min_value=200, max_value=2000, value=1000, step=100)
    sample_num = st.sidebar.slider("采样点数", min_value=200, max_value=3000, value=1000, step=100)

    df, pulse_times = generate_dbs_data(stim_freq, stim_amp, pulse_width, stim_strength, total_time, sample_num)
    title_text = f"核团膜电位时序曲线 | DBS频率：{stim_freq} Hz，幅值：{stim_amp} mV"
    st.success(f"✅ 已生成DBS仿真 | 频率：{stim_freq} Hz，幅值：{stim_amp} mV，脉宽：{pulse_width} ms")
else:
    upload_file = st.file_uploader("上传csv仿真数据文件", type=["csv"])
    if upload_file is not None:
        df = pd.read_csv(upload_file)
    else:
        st.info("⬆️ 请上传CSV格式仿真结果文件")


if df is not None:
    st.subheader("🎚️ 波形缩放调节（拖动滑块来放大缩小，搭配一键重置）")
    # 两个缩放滑块 + 重置按钮，解决缩放无效的问题
    col_reset, col_x, col_y = st.columns([1, 3, 3])
    with col_reset:
        reset_btn = st.button("🔄 恢复全局视图")
    with col_x:
        x_min, x_max = st.slider(
            "⏱️ 时间范围（ms）",
            min_value=float(df["time(ms)"].min()),
            max_value=float(df["time(ms)"].max()),
            value=(float(df["time(ms)"].min()), float(df["time(ms)"].max())) if not reset_btn else (float(df["time(ms)"].min()), float(df["time(ms)"].max())),
            step=1.0
        )
    with col_y:
        all_y = df[["STN","GPe","GPi"]]
        y_min_global = float(all_y.min().min())
        y_max_global = float(all_y.max().max())
        y_min, y_max = st.slider(
            "⚡ 膜电位范围（mV）",
            min_value=y_min_global,
            max_value=y_max_global,
            value=(y_min_global, y_max_global) if not reset_btn else (y_min_global, y_max_global),
            step=0.01
        )

    st.sidebar.header("📈 绘图设置")
    selected_cols = st.sidebar.multiselect(
        "选择需要绘制的核团信号",
        options=["STN", "GPe", "GPi"],
        default=["STN", "GPe", "GPi"]
    )

    if selected_cols:
        fig = px.line(
            df,
            x="time(ms)",
            y=selected_cols,
            title=title_text,
            labels={"value": "膜电位 (mV)", "time(ms)": "时间 (ms)", "variable": "核团"}
        )
        if mode == "内置仿真模型（可调DBS参数，直接演示）":
            for pt in pulse_times:
                fig.add_vline(x=pt, line_dash="dot", line_color="red", line_width=1.2)

        fig.update_layout(
            height=620,
            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max]
        )
        # 关闭容易误触的框选工具
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                'responsive': True,
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d','zoom2d']
            }
        )
        st.info("💡 使用方法：拖动时间滑块截取一小段时间，就实现放大；点【恢复全局视图】回到全貌！")
    else:
        st.warning("请至少勾选一个核团信号进行绘图")

# ========== 修改后的底部文字 ==========
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666666; font-size: 15px; line-height:1.6;'>"
    "🏫豫北医学院·王一浩 牛璐鸣 李雨轩<br>"
    "指导教师👩‍🏫：丰依凡"
    "</div>",
    unsafe_allow_html=True
)
