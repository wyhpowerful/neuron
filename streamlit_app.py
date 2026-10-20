import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import plotly.express as px
from streamlit.components.v1 import html

st.set_page_config(page_title="帕金森DBS神经动力学仿真可视化平台", layout="wide")

# ---------------------- 👀弹跳动画 ----------------------
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
st.markdown("📌 两种使用模式：①基底节环路网络仿真；②上传自己导出的CSV文件")

# ========== 【新增1：版本+更新日志板块】 ==========
with st.expander("📋 版本信息 & 更新日志", expanded=True):
    st.markdown("**当前版本：V1.2 | 更新日期：2026.10.20**")
    st.markdown("""
    - [2026.10.20] V1.2：新增【休闲窗口】，内置2048小游戏
    - [2026.10.20] V1.1：新增参数详情弹窗说明、参数原理解释模块
    - [2026.10.19] V1.0：完成基础三工况仿真、栅格图/LFP/时频图、CSV上传绘图功能
    """)
    st.info("💡 每次更新代码时，在这里直接修改版本号、日期、新增条目即可。")

# ---------------------- 默认配置 ----------------------
def get_default_config():
    return {
        "dt": 0.1,
        "sim_total_ms": 2000.0,
        "n_per_nuc": 40,
        "tau_exc": 3.0,
        "tau_inh": 6.0,
        "dbs_freq": 130,
        "dbs_amp": 60,
        "dbs_pulse_width": 0.1,
        "dbs_start_ms": 1000.0,
    }

# ---------------------- 参数说明字典【新增2：每个参数的说明，你后续可以继续扩充】 ----------------------
param_info = {
    "dt": {
        "brief": "仿真时间步长",
        "detail": "dt是数值仿真的时间间隔，单位ms。dt越小仿真精度越高，但计算速度更慢；dt太大容易出现微分方程发散、仿真失真。一般0.05~0.2ms比较合适。"
    },
    "sim_total_ms": {
        "brief": "仿真总时长",
        "detail": "整个仿真持续的总时间，单位ms。时长越长，能观察更长时间尺度的振荡节律，但仿真耗时线性增加。帕金森beta振荡一般观察1~3秒足够。"
    },
    "n_per_nuc": {
        "brief": "单核团神经元数量",
        "detail": "每个核团里面模拟多少个神经元。数量越多，群体放电统计效果越贴近真实环路，但是仿真计算速度明显变慢，建议新手20~40个。"
    },
    "tau_exc": {
        "brief": "兴奋性突触时间常数",
        "detail": "兴奋性突触电流衰减快慢。tau越大，兴奋信号持续时间越长，环路更容易产生同步振荡；兴奋性突触主要来自STN、皮层。"
    },
    "tau_inh": {
        "brief": "抑制性突触时间常数",
        "detail": "抑制性突触电流衰减快慢。抑制性突触来自GPe、纹状体，负责调控环路节律，是环路稳定的关键，和帕金森病理下抑制通路改变密切相关。"
    },
    "dbs_freq": {
        "brief": "DBS刺激频率",
        "detail": "深部脑刺激脉冲重复频率，临床治疗帕金森STN-DBS常规使用130Hz左右。低频刺激往往会加重beta振荡，高频才起到抑制病理性同步放电的效果。"
    },
    "dbs_amp": {
        "brief": "DBS刺激电流强度",
        "detail": "刺激电流幅值，决定对周围神经元的激活范围。电流太小无法调控环路，过大容易造成非目标区域激活，对应临床刺激副作用。"
    },
    "dbs_pulse_width": {
        "brief": "DBS脉冲宽度",
        "detail": "单个刺激脉冲持续时间，单位ms。临床常用脉宽多在0.06~0.12ms，脉宽决定电荷注入量，是DBS刺激的重要参数。"
    },
    "dbs_start_ms": {
        "brief": "DBS开启时刻",
        "detail": "仿真从什么时候开始施加DBS刺激。设置靠前可以全程刺激，设置靠后可以先观察帕金森自发振荡，再观察施加刺激之后的节律变化，方便前后对比。"
    }
}

# ---------------------- 神经元参数 ----------------------
neuron_param = {
    "Cortex":   {"a": 0.02, "b": 0.2,  "c": -65, "d": 8,  "v_rest": -70, "v_peak": 30, "I_bias": 10, "u0": 0},
    "Striatum": {"a": 0.01, "b": 0.03, "c": -65, "d": 15, "v_rest": -70, "v_peak": 30, "I_bias": 6,  "u0": 0},
    "GPe":      {"a": 0.025, "b": 0.01, "c": -65, "d": 12, "v_rest": -70, "v_peak": 30, "I_bias": 12, "u0": 0},
    "STN":      {"a": 0.01, "b": 0.03, "c": -65, "d": 15, "v_rest": -70, "v_peak": 30, "I_bias": 15, "u0": 0},
    "GPi":      {"a": 0.025, "b": 0.01, "c": -65, "d": 12, "v_rest": -70, "v_peak": 30, "I_bias": 12, "u0": 0},
    "Thalamus": {"a": 0.02, "b": 0.25, "c": -65, "d": 2,  "v_rest": -70, "v_peak": 30, "I_bias": 8,  "u0": 0},
}

# ---------------------- 网络拓扑 ----------------------
def create_sparse_connect(pre_nuc, post_nuc, n_pre, n_post, prob, weight, syn_type="exc"):
    rng = np.random.default_rng(42)
    mat = rng.choice([0, 1], size=(n_post, n_pre), p=[1 - prob, prob])
    return {
        "pre": pre_nuc,
        "post": post_nuc,
        "matrix": mat,
        "weight": weight,
        "syn_type": syn_type,
    }

def build_network_connectivity(n_per_nuc, pd_mode=False):
    conn = {}
    conn["Cortex_to_Striatum"] = create_sparse_connect("Cortex", "Striatum", n_per_nuc, n_per_nuc, 0.20, 1.0, "exc")
    conn["STN_to_GPe"]        = create_sparse_connect("STN", "GPe", n_per_nuc, n_per_nuc, 0.20, 1.2, "exc")
    conn["STN_to_GPi"]        = create_sparse_connect("STN", "GPi", n_per_nuc, n_per_nuc, 0.25, 1.5, "exc")
    conn["Thalamus_to_Cortex"]= create_sparse_connect("Thalamus", "Cortex", n_per_nuc, n_per_nuc, 0.20, 1.0, "exc")
    conn["Striatum_to_GPe"]   = create_sparse_connect("Striatum", "GPe", n_per_nuc, n_per_nuc, 0.20, -1.2, "inh")
    conn["Striatum_to_GPi"]   = create_sparse_connect("Striatum", "GPi", n_per_nuc, n_per_nuc, 0.20, -1.0, "inh")
    conn["GPe_to_STN"]        = create_sparse_connect("GPe", "STN", n_per_nuc, n_per_nuc, 0.25, -1.5, "inh")
    conn["GPe_to_GPi"]        = create_sparse_connect("GPe", "GPi", n_per_nuc, n_per_nuc, 0.20, -1.2, "inh")
    conn["GPi_to_Thalamus"]   = create_sparse_connect("GPi", "Thalamus", n_per_nuc, n_per_nuc, 0.20, -1.2, "inh")

    if pd_mode:
        conn["GPe_to_STN"]["weight"]   = -0.5
        conn["STN_to_GPi"]["weight"]   = 2.5
        conn["STN_to_GPe"]["weight"]   = 2.0
        conn["Striatum_to_GPi"]["weight"] = -0.6

    return conn

# ---------------------- 仿真核心 ----------------------
def run_simulation(config, neuron_param, conn, pd_mode=False, dbs_on=False):
    dt = config["dt"]
    sim_total_ms = config["sim_total_ms"]
    t_arr = np.arange(0, sim_total_ms, dt)
    Nt = len(t_arr)
    n_per_nuc = config["n_per_nuc"]
    tau_exc = config["tau_exc"]
    tau_inh = config["tau_inh"]
    dbs_freq = config["dbs_freq"]
    dbs_amp = config["dbs_amp"]
    dbs_start_t = config["dbs_start_ms"]
    dbs_pw = config.get("dbs_pulse_width", 0.1)
    dbs_period_ms = 1000.0 / dbs_freq

    nucleus_list = ["Cortex", "Striatum", "GPe", "STN", "GPi", "Thalamus"]

    v = {}
    u = {}
    I_bias = {}
    I_syn_exc = {}
    I_syn_inh = {}
    spk = {}
    spike_times = {}
    lfp = {}

    rng = np.random.default_rng(2026)

    for nuc in nucleus_list:
        p = neuron_param[nuc]
        v[nuc] = np.ones(n_per_nuc) * p["v_rest"]
        u[nuc] = np.ones(n_per_nuc) * p["u0"]
        I_bias[nuc] = p["I_bias"] + rng.uniform(-1.0, 1.0, n_per_nuc)
        I_syn_exc[nuc] = np.zeros(n_per_nuc)
        I_syn_inh[nuc] = np.zeros(n_per_nuc)
        spk[nuc] = np.zeros(n_per_nuc)
        spike_times[nuc] = [[] for _ in range(n_per_nuc)]
        lfp[nuc] = np.zeros(Nt)

    decay_exc = np.exp(-dt / tau_exc)
    decay_inh = np.exp(-dt / tau_inh)

    for key in conn:
        conn[key]["w_mat"] = conn[key]["matrix"] * conn[key]["weight"]

    trans = [
        ("Cortex", "Striatum", "exc"),
        ("STN", "GPe", "exc"),
        ("STN", "GPi", "exc"),
        ("Thalamus", "Cortex", "exc"),
        ("Striatum", "GPe", "inh"),
        ("Striatum", "GPi", "inh"),
        ("GPe", "STN", "inh"),
        ("GPe", "GPi", "inh"),
        ("GPi", "Thalamus", "inh"),
    ]

    for t_idx in range(Nt):
        t = t_arr[t_idx]

        for nuc in nucleus_list:
            I_syn_exc[nuc] *= decay_exc
            I_syn_inh[nuc] *= decay_inh

        for pre, post, st in trans:
            key = f"{pre}_to_{post}"
            inc = conn[key]["w_mat"].T @ spk[pre]
            if st == "exc":
                I_syn_exc[post] += inc
            else:
                I_syn_inh[post] += inc

        for nuc in nucleus_list:
            p = neuron_param[nuc]
            a, b, c, d, v_peak = p["a"], p["b"], p["c"], p["d"], p["v_peak"]

            I_total = I_bias[nuc] + I_syn_exc[nuc] + I_syn_inh[nuc]

            if nuc == "STN" and dbs_on and t >= dbs_start_t:
                phase = (t - dbs_start_t) % dbs_period_ms
                if phase < dbs_pw:
                    I_total = I_total + dbs_amp

            dv = 0.04 * v[nuc] ** 2 + 5 * v[nuc] + 140 - u[nuc] + I_total
            v[nuc] += dv * dt
            du = a * (b * v[nuc] - u[nuc])
            u[nuc] += du * dt

            spike_idx = v[nuc] >= v_peak
            spk[nuc][:] = 0

            if np.any(spike_idx):
                for idx in np.nonzero(spike_idx)[0]:
                    spike_times[nuc][idx].append(t)
                spk[nuc][spike_idx] = 1
                v[nuc][spike_idx] = c
                u[nuc][spike_idx] += d

        for nuc in nucleus_list:
            lfp[nuc][t_idx] = np.mean(v[nuc])

    return {
        "t": t_arr,
        "spike": spike_times,
        "lfp": lfp,
    }

# ---------------------- 绘图函数 ----------------------
plt.rcParams["axes.unicode_minus"] = False

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

# ---------------------- 三工况缓存仿真 ----------------------
@st.cache_data(max_entries=2)
def run_all_three_conditions(cfg):
    fs = 1000.0 / cfg["dt"]
    n = cfg["n_per_nuc"]

    conn_norm = build_network_connectivity(n, pd_mode=False)
    res_norm = run_simulation(cfg, neuron_param, conn_norm, pd_mode=False, dbs_on=False)

    conn_pd = build_network_connectivity(n, pd_mode=True)
    res_pd = run_simulation(cfg, neuron_param, conn_pd, pd_mode=True, dbs_on=False)

    conn_pddbs = build_network_connectivity(n, pd_mode=True)
    res_pddbs = run_simulation(cfg, neuron_param, conn_pddbs, pd_mode=True, dbs_on=True)

    return res_norm, res_pd, res_pddbs, fs

# ---------------------- 模式选择 ----------------------
mode = st.radio(
    "选择数据来源",
    ["基底节环路网络仿真", "上传自己的CSV文件"],
    horizontal=True,
)

df = None

# ---------------------- 模式1：环路仿真 + 侧边栏参数（带展开说明） ----------------------
if mode == "基底节环路网络仿真":
    st.sidebar.header("⚙️ 仿真参数设置")

    base_cfg = get_default_config()
    cfg = {}

    # ========== 侧边栏每一个参数，都带上展开详情 ==========
    with st.sidebar:
        st.markdown("#### 📐 基础仿真参数")
        with st.expander(f'dt — {param_info["dt"]["brief"]}'):
            st.info(param_info["dt"]["detail"])
        cfg["dt"] = st.number_input("仿真步长 dt(ms)", value=base_cfg["dt"], min_value=0.05, max_value=0.5, step=0.05)

        with st.expander(f'sim_total_ms — {param_info["sim_total_ms"]["brief"]}'):
            st.info(param_info["sim_total_ms"]["detail"])
        cfg["sim_total_ms"] = st.slider("仿真总时长(ms)", min_value=1000, max_value=3000, value=int(base_cfg["sim_total_ms"]), step=200)

        with st.expander(f'n_per_nuc — {param_info["n_per_nuc"]["brief"]}'):
            st.info(param_info["n_per_nuc"]["detail"])
        cfg["n_per_nuc"] = st.slider("单核团神经元数量", min_value=10, max_value=60, value=base_cfg["n_per_nuc"], step=5)

        st.markdown("#### 🧠 突触动力学参数")
        with st.expander(f'tau_exc — {param_info["tau_exc"]["brief"]}'):
            st.info(param_info["tau_exc"]["detail"])
        cfg["tau_exc"] = st.slider("兴奋性时间常数(ms)", 1.0, 8.0, base_cfg["tau_exc"], 0.5)

        with st.expander(f'tau_inh — {param_info["tau_inh"]["brief"]}'):
            st.info(param_info["tau_inh"]["detail"])
        cfg["tau_inh"] = st.slider("抑制性时间常数(ms)", 2.0, 12.0, base_cfg["tau_inh"], 0.5)

        st.markdown("#### ⚡ DBS刺激参数")
        with st.expander(f'dbs_freq — {param_info["dbs_freq"]["brief"]}'):
            st.info(param_info["dbs_freq"]["detail"])
        cfg["dbs_freq"] = st.slider("DBS频率(Hz)", 20, 180, base_cfg["dbs_freq"], 5)

        with st.expander(f'dbs_amp — {param_info["dbs_amp"]["brief"]}'):
            st.info(param_info["dbs_amp"]["detail"])
        cfg["dbs_amp"] = st.slider("DBS刺激电流", 10, 120, base_cfg["dbs_amp"], 5)

        with st.expander(f'dbs_pulse_width — {param_info["dbs_pulse_width"]["brief"]}'):
            st.info(param_info["dbs_pulse_width"]["detail"])
        cfg["dbs_pulse_width"] = st.number_input("DBS脉宽(ms)", value=base_cfg["dbs_pulse_width"], min_value=0.05, max_value=1.0, step=0.05)

        with st.expander(f'dbs_start_ms — {param_info["dbs_start_ms"]["brief"]}'):
            st.info(param_info["dbs_start_ms"]["detail"])
        cfg["dbs_start_ms"] = st.slider("DBS开启时刻(ms)", 200, 2000, int(base_cfg["dbs_start_ms"]), 100)

    run_btn = st.button("▶ 运行三工况仿真（正常 / PD / PD+DBS）")

    if run_btn:
        with st.spinner("正在仿真计算，请等待，神经元数量越大耗时越长……"):
            res_norm, res_pd, res_pddbs, fs = run_all_three_conditions(cfg)

            t_norm = res_norm["t"]
            t_pd = res_pd["t"]
            t_pddbs = res_pddbs["t"]

            fig, axes = plt.subplots(3, 3, figsize=(16, 12))

            plot_raster(axes[0, 0], t_norm, res_norm["spike"]["GPi"], title="正常 GPi 放电栅格图")
            plot_lfp(axes[0, 1], t_norm, res_norm["lfp"]["GPi"], title="正常 GPi LFP")
            plot_stft(axes[0, 2], t_norm, res_norm["lfp"]["GPi"], fs=fs, title="正常 GPi 时频图")

            plot_raster(axes[1, 0], t_pd, res_pd["spike"]["GPi"], title="PD患病 GPi 放电栅格图")
            plot_lfp(axes[1, 1], t_pd, res_pd["lfp"]["GPi"], title="PD患病 GPi LFP")
            plot_stft(axes[1, 2], t_pd, res_pd["lfp"]["GPi"], fs=fs, title="PD患病 GPi 时频图")

            plot_raster(axes[2, 0], t_pddbs, res_pddbs["spike"]["GPi"], title="PD+DBS刺激 GPi 放电栅格图")
            plot_lfp(axes[2, 1], t_pddbs, res_pddbs["lfp"]["GPi"], title="PD+DBS刺激 GPi LFP")
            plot_stft(axes[2, 2], t_pddbs, res_pddbs["lfp"]["GPi"], fs=fs, title="PD+DBS刺激 GPi 时频图")

            fig.tight_layout()
            st.pyplot(fig)

            st.success("✅仿真完成！上图为：放电栅格、LFP局部场电位、时频分析")

# ---------------------- 模式2：CSV上传 ----------------------
else:
    upload_file = st.file_uploader("上传csv仿真数据文件", type=["csv"])

    if upload_file is not None:
        df = pd.read_csv(upload_file)
    else:
        st.info("⬆️ 请上传CSV格式仿真结果文件")

# ---------------------- CSV模式绘图与缩放滑块 ----------------------
if df is not None:
    st.subheader("🎚️ 波形缩放调节（拖动滑块来放大缩小，搭配一键重置）")

    col_reset, col_x, col_y = st.columns([1, 3, 3])

    with col_reset:
        reset_btn = st.button("🔄 恢复全局视图")

    with col_x:
        x_min, x_max = st.slider(
            "⏱️ 时间范围（ms）",
            min_value=float(df["time(ms)"].min()),
            max_value=float(df["time(ms)"].max()),
            value=(float(df["time(ms)"].min()), float(df["time(ms)"].max())) if not reset_btn else (float(df["time(ms)"].min()), float(df["time(ms)"].max())),
            step=1.0,
        )

    with col_y:
        all_y = df[["STN", "GPe", "GPi"]]
        y_min_global = float(all_y.min().min())
        y_max_global = float(all_y.max().max())

        y_min, y_max = st.slider(
            "⚡ 膜电位范围（mV）",
            min_value=y_min_global,
            max_value=y_max_global,
            value=(y_min_global, y_max_global) if not reset_btn else (y_min_global, y_max_global),
            step=0.01,
        )

    st.sidebar.header("📈 绘图设置")

    selected_cols = st.sidebar.multiselect(
        "选择需要绘制的核团信号",
        options=["STN", "GPe", "GPi"],
        default=["STN", "GPe", "GPi"],
    )

    if selected_cols:
        fig = px.line(
            df,
            x="time(ms)",
            y=selected_cols,
            title="核团膜电位时序曲线",
            labels={"value": "膜电位 (mV)", "time(ms)": "时间 (ms)", "variable": "核团"},
        )

        fig.update_layout(
            height=620,
            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "responsive": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d", "zoom2d"],
            },
        )

        st.info("💡 使用方法：拖动时间滑块截取一小段时间，就实现放大；点【恢复全局视图】回到全貌！")
    else:
        st.warning("请至少勾选一个核团信号进行绘图")


# ========== 🎮 【升级版本：点击按钮控制2048，手机友好版】 ==========
with st.expander("🎮 休闲窗口 | 仿真累了可以玩一会2048", expanded=False):
    st.markdown("点击下方方向按钮控制方块移动，合并相同数字，合成2048即胜利！")
    game_html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.">
<style>
*{box-sizing: border-box}
body{margin:0;padding:5px}
.game-container{width:340px;margin:10px auto;font-family:Arial}
.grid{display:grid;grid-template-columns: repeat(4, 1fr);gap:8px;background:#bbada0;padding:8px;border-radius:6px}
.cell{width:70px;height:70px;background:#cdc1b4;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:bold;color:#776e65}
.t2{background:#eee4da}
.t4{background:#ede0c8}
.t8{background:#f2b179;color:white}
.t16{background:#f59563;color:white}
.t32{background:#f67c5f;color:white}
.t64{background:#f65e3b;color:white}
.t128{background:#edcf72;color:white;font-size:20px}
.t256{background:#edcc61;color:white;font-size:20px}
.t512{background:#edc850;color:white;font-size:20px}
.t1024{background:#edc53f;color:white;font-size:16px}
.t2048{background:#edc22e;color:white;font-size:16px}
.control{margin-top:15px;text-align:center}
.btn{width:70px;height:50px;margin:4px;font-size:20px;border:none;border-radius:8px;background:#d9d2c6;color:#444;font-weight:bold}
.btn:active{background:#b9b09f}
</style>
</head>
<body>
<div class="game-container">
<div class="grid" id="grid"></div>
<p id="tip" style="text-align:center;font-size:20px">分数：0</p>
<div class="control">
<div><button class="btn" onclick="move('up')">↑</button></div>
<div>
<button class="btn" onclick="move('left')">←</button>
<button class="btn" onclick="move('down')">↓</button>
<button class="btn" onclick="move('right')">→</button>
</div>
</div>
</div>
<script>
const gridEl = document.getElementById("grid");
const tipEl = document.getElementById("tip");
let grid = Array(4).fill().map(()=>Array(4).fill(0));
let score = 0;
function randEmpty(){
    let arr=[];
    for(let i=0;i<4;i++){
        for(let j=0;j<4;j++){
            if(grid[i][j]===0){
                arr.push([i,j]);
            }
        }
    }
    if(arr.length===0) return null;
    return arr[Math.floor(Math.random()*arr.length)];
}
function addBlock(){
    let p=randEmpty();
    if(!p) return;
    grid[p[0]][p[1]] = Math.random()>0.1?2:4;
}
function render(){
    gridEl.innerHTML="";
    for(let i=0;i<4;i++){
        for(let j=0;j<4;j++){
            let v=grid[i][j];
            let d=document.createElement("div");
            d.className="cell "+(v?"t"+v:"");
            d.innerText=v||"";
            gridEl.appendChild(d);
        }
    }
    tipEl.innerText="分数："+score;
}
function slide(row){
    let a=row.filter(x=>x!==0);
    for(let i=0;i<a.length-1;i++){
        if(a[i]==a[i+1]){
            a[i]*=2;
            score += a[i];
            a.splice(i+1,1);
        }
    }
    while(a.length<4) a.push(0);
    return a;
}
function move(dir){
    let old = JSON.stringify(grid);
    if(dir=="left") for(let i=0;i<4;i++) grid[i]=slide(grid[i]);
    if(dir=="right") for(let i=0;i<4;i++) grid[i]=slide(grid[i].reverse()).reverse();
    if(dir=="up"){
        for(let j=0;j<4;j++){
            let c=[grid[0][j],grid[1][j],grid[2][j],grid[3][j]];
            c=slide(c);
            for(let i=0;i<4;i++) grid[i][j]=c[i];
        }
    }
    if(dir=="down"){
        for(let j=0;j<4;j++){
            let c=[grid[0][j],grid[1][j],grid[2][j],grid[3][j]];
            c=slide(c.reverse()).reverse();
            for(let i=0;i<4;i++) grid[i][j]=c[i];
        }
    }
    if(JSON.stringify(grid)!==old) addBlock();
    render();
}
// 保留电脑方向键兼容
document.onkeydown = e=>{
    if(e.key=="ArrowUp"){e.preventDefault();move("up");}
    if(e.key=="ArrowDown"){e.preventDefault();move("down");}
    if(e.key=="ArrowLeft"){e.preventDefault();move("left");}
    if(e.key=="ArrowRight"){e.preventDefault();move("right");}
}
addBlock();addBlock();render();
</script>
</body>
</html>
    """
    html(game_html, height=580)
# ---------------------- 底部署名 ----------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666666; font-size: 15px; line-height:1.6;'>"
    "🏫豫北医学院·王一浩 牛璐鸣 李雨轩<br>"
    "指导教师👩‍🏫：丰依凡"
    "</div>",
    unsafe_allow_html=True,
)
