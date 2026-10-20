import streamlit as st
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

st.set_page_config(page_title="帕金森DBS神经动力学仿真可视化平台", layout="wide")

# 动画标题样式
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
</style>
<span class="eye1">✨</span>
<span class="eye2">✨</span> 两种使用模式：①基底节环路网络仿真；②上传自己导出的CSV文件
"""
st.markdown(bounce_html, unsafe_allow_html=True)


# ========== 绘图函数 ==========
def plot_raster(ax, t, spike_rec, title=""):
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("神经元编号")
    ax.set_title(title)
    if len(spike_rec) == 0:
        return
    for neuron_idx, spike_times in enumerate(spike_rec):
        stimes = np.asarray(spike_times)
        if stimes.size > 0:
            ax.plot(stimes, np.full(stimes.size, neuron_idx), 'k.', markersize=1)
    ax.set_xlim(t[0], t[-1])


def plot_lfp(ax, t, lfp, title=""):
    ax.plot(t, lfp, color="#1f77b4", linewidth=1.0)
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("LFP")
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])


def plot_stft(ax, t, lfp, fs, title=""):
    f, t_stft, Zxx = signal.stft(lfp, fs=fs, nperseg=2048, noverlap=1792)
    im = ax.pcolormesh(t_stft * 1000, f, np.abs(Zxx), shading='gouraud', cmap="jet")
    ax.set_ylim(0, 60)
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("频率 (Hz)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return im


# ========== 仿真模型函数 ==========
def run_model(condition="normal"):
    """简化基底节STN-GPe-GPi环路仿真，返回spike、lfp、时间轴"""
    t_total = 2000
    dt = 1
    t = np.arange(0, t_total, dt)
    n_gpi = 20
    fs = 1000

    # 基础振荡参数
    if condition == "normal":
        freq = 20
        amp = 0.3
        burst_prob = 0.002
    elif condition == "pd":
        freq = 13
        amp = 0.8
        burst_prob = 0.008
    else:  # dbs
        freq = 22
        amp = 0.25
        burst_prob = 0.0015

    # LFP信号
    lfp_raw = amp * np.sin(2 * np.pi * freq * t / fs) + 0.15 * np.random.randn(len(t))

    # 生成放电序列
    spike_gpi = [[] for _ in range(n_gpi)]
    np.random.seed(42)
    for nt, ti in enumerate(t):
        if np.random.rand() < burst_prob:
            n_sp = np.random.randint(2,6)
            for _ in range(n_sp):
                nid = np.random.randint(0, n_gpi)
                spike_gpi[nid].append(ti)

    out = {
        "t": t,
        "spike": {
            "GPi": spike_gpi
        },
        "lfp": {
            "GPi": lfp_raw
        }
    }
    return out


# ========== 主界面逻辑 ==========
mode = st.radio("选择数据来源", ["基底节环路网络仿真", "上传自己的CSV文件"])

if mode == "基底节环路网络仿真":
    if st.button("▶ 运行三工况仿真（正常 / PD / PD+DBS）"):
        with st.spinner("仿真计算中，请稍等..."):
            res_norm = run_model(condition="normal")
            res_pd = run_model(condition="pd")
            res_pddbs = run_model(condition="dbs")
            t = res_norm["t"]
            fs = 1000

            fig, axes = plt.subplots(3, 3, figsize=(18, 12))

            # 第一行：正常
            plot_raster(axes[0,0], t, res_norm["spike"]["GPi"], title="正常 GPi 放电栅格图")
            plot_lfp(axes[0,1], t, res_norm["lfp"]["GPi"], title="正常 GPi LFP")
            plot_stft(axes[0,2], t, res_norm["lfp"]["GPi"], fs=fs, title="正常 GPi 时频图")

            # 第二行：PD
            plot_raster(axes[1,0], t, res_pd["spike"]["GPi"], title="PD患病 GPi 放电栅格图")
            plot_lfp(axes[1,1], t, res_pd["lfp"]["GPi"], title="PD患病 GPi LFP")
            plot_stft(axes[1,2], t, res_pd["lfp"]["GPi"], fs=fs, title="PD患病 GPi 时频图")

            # 第三行：PD+DBS
            plot_raster(axes[2,0], t, res_pddbs["spike"]["GPi"], title="PD+DBS刺激 GPi 放电栅格图")
            plot_lfp(axes[2,1], t, res_pddbs["lfp"]["GPi"], title="PD+DBS刺激 GPi LFP")
            plot_stft(axes[2,2], t, res_pddbs["lfp"]["GPi"], fs=fs, title="PD+DBS刺激 GPi 时频图")

            fig.tight_layout()
            st.pyplot(fig)

else:
    up_file = st.file_uploader("上传CSV文件", type="csv")
    if up_file is not None:
        import pandas as pd
        df = pd.read_csv(up_file)
        st.dataframe(df)
        st.info("CSV读取完成，你后续可以在这里加绘图逻辑")
