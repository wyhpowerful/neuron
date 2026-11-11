import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import plotly.express as px
from streamlit.components.v1 import html

# ====================== 页面全局配置｜Apple极简风格 ======================
st.set_page_config(
    page_title="帕金森DBS神经动力学仿真可视化平台",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Apple风格全局CSS - 极致重构版
global_css = """
<style>
/* 隐藏 Streamlit 原生的顶部彩虹条和右侧工具栏 */
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
header { background: transparent !important; }

/* 全局基础字体，模拟苹果 San Francisco 字体 */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #ffffff;
    color: #1d1d1f;
    -webkit-font-smoothing: antialiased;
}

/* 主页面容器：大留白，限制最大宽度 */
.block-container {
    padding-top: 4rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* 苹果风大标题 */
.app-h1 {
    font-size: 56px;
    font-weight: 700;
    letter-spacing: -0.015em;
    text-align: center;
    margin-bottom: 10px;
    color: #1d1d1f;
}
.app-subtitle {
    font-size: 24px;
    font-weight: 400;
    letter-spacing: 0.009em;
    text-align: center;
    color: #86868b;
    margin-bottom: 50px;
}

/* 顶部卡片：极致扁平，淡阴影，超大圆角 */
div.app-card {
    background-color: #ffffff;
    border-radius: 24px;
    padding: 2rem 2.5rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
    margin-bottom: 30px;
    border: none;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
div.app-card:hover {
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    transform: translateY(-2px);
}

/* 团队成员按钮：苹果官网同款浅灰胶囊按钮 */
.app-team-btn > button {
    width: 100%;
    height: 4rem;
    border-radius: 999px !important; /* 胶囊形状 */
    font-size: 17px;
    font-weight: 500;
    border: none;
    background: #f5f5f7;
    color: #1d1d1f !important;
    transition: all 0.2s ease;
}
.app-team-btn > button:hover {
    background: #e8e8ed;
    transform: scale(1.02);
}

/* 主运行按钮：苹果蓝胶囊 */
.stButton > button {
    width: 100%;
    border-radius: 999px !important;
    font-weight: 500;
    font-size: 17px;
    height: 3.5rem;
    background: #0071e3; /* 苹果官网蓝 */
    color: #ffffff !important;
    border: none;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #0077ed;
    transform: scale(1.01);
}

/* 侧边栏：极简白色，去边框 */
[data-testid="stSidebar"] {
    background-color: #fbfbfd !important;
    border-right: 1px solid #f5f5f7;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-weight: 600;
    color: #1d1d1f;
    letter-spacing: -0.01em;
}

/* 折叠面板：去边框 */
[data-testid="stExpander"] details summary {
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    background-color: #f5f5f7;
    border: none !important;
    font-weight: 500;
}
[data-testid="stExpander"] details:hover summary {
    background-color: #e8e8ed;
}

/* 隐藏所有原生横线 */
hr { display: none !important; }

/* 页脚卡片 */
div.app-footer {
    background-color: #f5f5f7;
    border-radius: 24px;
    padding: 30px;
    text-align: center;
    margin-top: 50px;
    color: #86868b;
    font-size: 14px;
}
</style>
"""
st.markdown(global_css, unsafe_allow_html=True)

# ========== 页面大标题｜苹果官网同款排版 ==========
st.markdown("""
<div class="app-h1">帕金森DBS神经动力学仿真平台</div>
<div class="app-subtitle">基底节环路 Izhikevich 神经元网络 · 深部脑刺激数值仿真系统</div>
""", unsafe_allow_html=True)

# ==========【免责声明卡片】 ==========
st.markdown('<div class="app-card">', unsafe_allow_html=True)
with st.expander("⚠️ 免责声明", expanded=False):
    st.warning("""
本平台仅为生物医学工程课程学习、科研教学仿真演示工具。
1. 所有仿真数据、模型结果均为理想化数值模拟，**不能替代临床医学诊断、治疗方案参考，不可用于实际病人诊疗**。
2. 模型参数、环路连接为简化理论模型，与真实人体大脑存在差距。
3. 若用于报告、作业引用，请自行核对文献资料；项目作者不对使用本工具产生的任何直接或间接后果承担责任。
4. 本项目仅供非盈利学习交流使用，禁止商用。
""")
st.markdown('</div>', unsafe_allow_html=True)

# ========== 【版本+更新日志卡片】 ==========
st.markdown('<div class="app-card">', unsafe_allow_html=True)
with st.expander("📋 版本信息 & 更新日志", expanded=False):
    st.markdown("**当前版本：V2.0｜Apple极简UI｜更新日期：2026.11.11**")
    st.markdown("""
    - [2026.11.11] V2.0：UI 侧边栏紧凑化；三工况改为 Tab 展示；弹球游戏修复无限弹并新增关卡；新增黄金矿工游戏
    - [2026.10.20] V1.7：重构Apple官网简约风格；移除所有横杠与粗侧边色条；优化卡片悬浮交互
    - [2026.10.20] V1.5：新增弹球打砖块休闲小游戏
    - [2026.10.19] V1.0：完成基础三工况仿真、栅格图/LFP/时频图、CSV上传绘图功能
    """)
st.markdown('</div>', unsafe_allow_html=True)

# ========== ✨【项目团队成员模块】 ==========
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.markdown('<div class="app-section-title">👥 项目团队成员（点击名字查看简介）</div>', unsafe_allow_html=True)

team_data = [
    {"name": "王一浩", "intro": "**项目负责人**\n负责代码撰写，结课汇报。"},
    {"name": "牛璐鸣", "intro": "**入党积极分子**\n负责论文整合，撰写，协调。"},
    {"name": "李雨轩", "intro": "**入党积极分子**\n负责论文矫正，论文润色，纠错。"},
    {"name": "丰依凡", "intro": "**项目带头人 | 硕士生导师**\n现工作于豫北医学院，项目奠基人。"}
]


@st.dialog("成员简介")
def show_member_dialog(member_name, member_intro):
    st.subheader(member_name)
    st.markdown(member_intro)


col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]

for idx, member in enumerate(team_data):
    with cols[idx]:
        st.markdown('<div class="app-team-btn">', unsafe_allow_html=True)
        if st.button(member["name"], key=f"team_btn_{idx}"):
            show_member_dialog(member["name"], member["intro"])
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ---------------------- 默认配置 ----------------------
def get_default_config():
    return {
        "dt": 0.1, "sim_total_ms": 2000.0, "n_per_nuc": 40,
        "tau_exc": 3.0, "tau_inh": 6.0,
        "dbs_freq": 130, "dbs_amp": 60,
        "dbs_pulse_width": 0.1, "dbs_start_ms": 1000.0,
    }


param_info = {
    "dt": {"brief": "仿真时间步长",
           "detail": "dt是数值仿真的时间间隔，单位ms。dt越小仿真精度越高，但计算速度更慢；dt太大容易出现微分方程发散、仿真失真。一般0.05~0.2ms比较合适。"},
    "sim_total_ms": {"brief": "仿真总时长",
                     "detail": "整个仿真持续的总时间，单位ms。时长越长，能观察更长时间尺度的振荡节律，但仿真耗时线性增加。帕金森beta振荡一般观察1~3秒足够。"},
    "n_per_nuc": {"brief": "单核团神经元数量",
                  "detail": "每个核团里面模拟多少个神经元。数量越多，群体放电统计效果越贴近真实环路，但是仿真计算速度明显变慢，建议新手20~40个。"},
    "tau_exc": {"brief": "兴奋性突触时间常数",
                "detail": "兴奋性突触电流衰减快慢。tau越大，兴奋信号持续时间越长，环路更容易产生同步振荡；兴奋性突触主要来自STN、皮层。"},
    "tau_inh": {"brief": "抑制性突触时间常数",
                "detail": "抑制性突触电流衰减快慢。抑制性突触来自GPe、纹状体，负责调控环路节律，是环路稳定的关键，和帕金森病理下抑制通路改变密切相关。"},
    "dbs_freq": {"brief": "DBS刺激频率",
                 "detail": "深部脑刺激脉冲重复频率，临床治疗帕金森STN‑DBS常规使用130Hz左右。低频刺激往往会加重beta振荡，高频才起到抑制病理性同步放电的效果。"},
    "dbs_amp": {"brief": "DBS刺激电流强度",
                "detail": "刺激电流幅值，决定对周围神经元的激活范围。电流太小无法调控环路，过大容易造成非目标区域激活，对应临床刺激副作用。"},
    "dbs_pulse_width": {"brief": "DBS脉冲宽度",
                        "detail": "单个刺激脉冲持续时间，单位ms。临床常用脉宽多在0.06~0.12ms，脉宽决定电荷注入量，是DBS刺激的重要参数。"},
    "dbs_start_ms": {"brief": "DBS开启时刻",
                     "detail": "仿真从什么时候开始施加DBS刺激。设置靠前可以全程刺激，设置靠后可以先观察帕金森自发振荡，再观察施加刺激之后的节律变化，方便前后对比。"},
}

neuron_param = {
    "Cortex": {"a": 0.02, "b": 0.2, "c": -65, "d": 8, "v_rest": -70, "v_peak": 30, "I_bias": 10, "u0": 0},
    "Striatum": {"a": 0.01, "b": 0.03, "c": -65, "d": 15, "v_rest": -70, "v_peak": 30, "I_bias": 6, "u0": 0},
    "GPe": {"a": 0.025, "b": 0.01, "c": -65, "d": 12, "v_rest": -70, "v_peak": 30, "I_bias": 12, "u0": 0},
    "STN": {"a": 0.01, "b": 0.03, "c": -65, "d": 15, "v_rest": -70, "v_peak": 30, "I_bias": 15, "u0": 0},
    "GPi": {"a": 0.025, "b": 0.01, "c": -65, "d": 12, "v_rest": -70, "v_peak": 30, "I_bias": 12, "u0": 0},
    "Thalamus": {"a": 0.02, "b": 0.25, "c": -65, "d": 2, "v_rest": -70, "v_peak": 30, "I_bias": 8, "u0": 0},
}


def create_sparse_connect(pre_nuc, post_nuc, n_pre, n_post, prob, weight, syn_type="exc"):
    rng = np.random.default_rng(42)
    mat = rng.choice([0, 1], size=(n_post, n_pre), p=[1 - prob, prob])
    return {"pre": pre_nuc, "post": post_nuc, "matrix": mat, "weight": weight, "syn_type": syn_type}


def build_network_connectivity(n_per_nuc, pd_mode=False):
    conn = {}
    conn["Cortex_to_Striatum"] = create_sparse_connect("Cortex", "Striatum", n_per_nuc, n_per_nuc, 0.20, 1.0, "exc")
    conn["STN_to_GPe"] = create_sparse_connect("STN", "GPe", n_per_nuc, n_per_nuc, 0.20, 1.2, "exc")
    conn["STN_to_GPi"] = create_sparse_connect("STN", "GPi", n_per_nuc, n_per_nuc, 0.25, 1.5, "exc")
    conn["Thalamus_to_Cortex"] = create_sparse_connect("Thalamus", "Cortex", n_per_nuc, n_per_nuc, 0.20, 1.0, "exc")
    conn["Striatum_to_GPe"] = create_sparse_connect("Striatum", "GPe", n_per_nuc, n_per_nuc, 0.20, -1.2, "inh")
    conn["Striatum_to_GPi"] = create_sparse_connect("Striatum", "GPi", n_per_nuc, n_per_nuc, 0.20, -1.0, "inh")
    conn["GPe_to_STN"] = create_sparse_connect("GPe", "STN", n_per_nuc, n_per_nuc, 0.25, -1.5, "inh")
    conn["GPe_to_GPi"] = create_sparse_connect("GPe", "GPi", n_per_nuc, n_per_nuc, 0.20, -1.2, "inh")
    conn["GPi_to_Thalamus"] = create_sparse_connect("GPi", "Thalamus", n_per_nuc, n_per_nuc, 0.20, -1.2, "inh")
    if pd_mode:
        conn["GPe_to_STN"]["weight"] = -0.5
        conn["STN_to_GPi"]["weight"] = 2.5
        conn["STN_to_GPe"]["weight"] = 2.0
        conn["Striatum_to_GPi"]["weight"] = -0.6
    return conn


def run_simulation(config, neuron_param, conn, pd_mode=False, dbs_on=False):
    dt = config["dt"];
    sim_total_ms = config["sim_total_ms"]
    t_arr = np.arange(0, sim_total_ms, dt);
    Nt = len(t_arr)
    n_per_nuc = config["n_per_nuc"]
    tau_exc = config["tau_exc"];
    tau_inh = config["tau_inh"]
    dbs_freq = config["dbs_freq"];
    dbs_amp = config["dbs_amp"]
    dbs_start_t = config["dbs_start_ms"]
    dbs_pw = config.get("dbs_pulse_width", 0.1)
    dbs_period_ms = 1000.0 / dbs_freq
    nucleus_list = ["Cortex", "Striatum", "GPe", "STN", "GPi", "Thalamus"]
    v = {};
    u = {};
    I_bias = {};
    I_syn_exc = {};
    I_syn_inh = {};
    spk = {};
    spike_times = {};
    lfp = {}
    rng = np.random.default_rng(2026)
    for nuc in nucleus_list:
        p = neuron_param[nuc]
        v[nuc] = np.ones(n_per_nuc) * p["v_rest"];
        u[nuc] = np.ones(n_per_nuc) * p["u0"]
        I_bias[nuc] = p["I_bias"] + rng.uniform(-1.0, 1.0, n_per_nuc)
        I_syn_exc[nuc] = np.zeros(n_per_nuc);
        I_syn_inh[nuc] = np.zeros(n_per_nuc)
        spk[nuc] = np.zeros(n_per_nuc);
        spike_times[nuc] = [[] for _ in range(n_per_nuc)];
        lfp[nuc] = np.zeros(Nt)
    decay_exc = np.exp(-dt / tau_exc);
    decay_inh = np.exp(-dt / tau_inh)
    for key in conn: conn[key]["w_mat"] = conn[key]["matrix"] * conn[key]["weight"]
    trans = [("Cortex", "Striatum", "exc"), ("STN", "GPe", "exc"), ("STN", "GPi", "exc"),
             ("Thalamus", "Cortex", "exc"), ("Striatum", "GPe", "inh"), ("Striatum", "GPi", "inh"),
             ("GPe", "STN", "inh"), ("GPe", "GPi", "inh"), ("GPi", "Thalamus", "inh")]
    for t_idx in range(Nt):
        t = t_arr[t_idx]
        for nuc in nucleus_list:
            I_syn_exc[nuc] *= decay_exc;
            I_syn_inh[nuc] *= decay_inh
        for pre, post, st in trans:
            key = f"{pre}_to_{post}"
            inc = conn[key]["w_mat"].T @ spk[pre]
            if st == "exc":
                I_syn_exc[post] += inc
            else:
                I_syn_inh[post] += inc
        for nuc in nucleus_list:
            p = neuron_param[nuc];
            a, b, c, d, v_peak = p["a"], p["b"], p["c"], p["d"], p["v_peak"]
            I_total = I_bias[nuc] + I_syn_exc[nuc] + I_syn_inh[nuc]
            if nuc == "STN" and dbs_on and t >= dbs_start_t:
                phase = (t - dbs_start_t) % dbs_period_ms
                if phase < dbs_pw: I_total = I_total + dbs_amp
            dv = 0.04 * v[nuc] ** 2 + 5 * v[nuc] + 140 - u[nuc] + I_total
            v[nuc] += dv * dt
            du = a * (b * v[nuc] - u[nuc]);
            u[nuc] += du * dt
            spike_idx = v[nuc] >= v_peak;
            spk[nuc][:] = 0
            if np.any(spike_idx):
                for idx in np.nonzero(spike_idx)[0]: spike_times[nuc][idx].append(t)
                spk[nuc][spike_idx] = 1;
                v[nuc][spike_idx] = c;
                u[nuc][spike_idx] += d
        for nuc in nucleus_list: lfp[nuc][t_idx] = np.mean(v[nuc])
    return {"t": t_arr, "spike": spike_times, "lfp": lfp}


plt.rcParams["axes.unicode_minus"] = False


def plot_raster(ax, t, spike_rec, title=""):
    ax.set_xlabel("时间 (ms)");
    ax.set_ylabel("神经元编号");
    ax.set_title(title)
    if len(spike_rec) == 0: return
    for i, st_ in enumerate(spike_rec):
        s = np.asarray(st_)
        if s.size > 0: ax.plot(s, np.full(s.size, i), 'k.', markersize=1)
    ax.set_xlim(t[0], t[-1])


def plot_lfp(ax, t, lfp, title=""):
    ax.plot(t, lfp, color="#1f77b4", linewidth=1.0)
    ax.set_xlabel("时间 (ms)");
    ax.set_ylabel("LFP");
    ax.set_title(title);
    ax.set_xlim(t[0], t[-1])


def plot_stft(ax, t, lfp, fs, title=""):
    f, t_stft, Zxx = signal.stft(lfp, fs=fs, nperseg=2048, noverlap=1792)
    im = ax.pcolormesh(t_stft * 1000, f, np.abs(Zxx), shading='gouraud', cmap="jet")
    ax.set_ylim(0, 60);
    ax.set_xlabel("时间 (ms)");
    ax.set_ylabel("频率 (Hz)");
    ax.set_title(title)
    plt.colorbar(im, ax=ax);
    return im


@st.cache_data(max_entries=2)
def run_all_three_conditions(cfg):
    fs = 1000.0 / cfg["dt"];
    n = cfg["n_per_nuc"]
    conn_norm = build_network_connectivity(n, pd_mode=False)
    res_norm = run_simulation(cfg, neuron_param, conn_norm, pd_mode=False, dbs_on=False)
    conn_pd = build_network_connectivity(n, pd_mode=True)
    res_pd = run_simulation(cfg, neuron_param, conn_pd, pd_mode=True, dbs_on=False)
    conn_pddbs = build_network_connectivity(n, pd_mode=True)
    res_pddbs = run_simulation(cfg, neuron_param, conn_pddbs, pd_mode=True, dbs_on=True)
    return res_norm, res_pd, res_pddbs, fs


# ====================== 模式选择 ======================
mode = st.radio("选择数据来源", ["基底节环路网络仿真", "上传自己的CSV文件"], horizontal=True)
df = None


# ---------------------- 侧边栏：紧凑参数面板 ----------------------
def param_row(key, label, widget_fn):
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(f"**{label}**")
        st.caption(param_info[key]["brief"])
    with c2:
        with st.popover("❓"):
            st.markdown(f"**{param_info[key]['brief']}**")
            st.markdown(param_info[key]["detail"])
    return widget_fn()


if mode == "基底节环路网络仿真":
    st.sidebar.header("⚙️ 仿真参数设置")
    base_cfg = get_default_config()
    cfg = {}
    with st.sidebar:
        st.markdown("#### 📐 基础仿真参数")
        cfg["dt"] = param_row("dt", "仿真步长 dt (ms)",
                              lambda: st.number_input("dt", value=base_cfg["dt"], min_value=0.05, max_value=0.5,
                                                      step=0.05, label_visibility="collapsed"))
        cfg["sim_total_ms"] = param_row("sim_total_ms", "仿真总时长 (ms)",
                                        lambda: st.slider("sim_total_ms", 1000, 3000, int(base_cfg["sim_total_ms"]),
                                                          200, label_visibility="collapsed"))
        cfg["n_per_nuc"] = param_row("n_per_nuc", "单核团神经元数量",
                                     lambda: st.slider("n_per_nuc", 10, 60, base_cfg["n_per_nuc"], 5,
                                                       label_visibility="collapsed"))

        st.markdown("#### 🧠 突触动力学参数")
        cfg["tau_exc"] = param_row("tau_exc", "兴奋性时间常数 (ms)",
                                   lambda: st.slider("tau_exc", 1.0, 8.0, base_cfg["tau_exc"], 0.5,
                                                     label_visibility="collapsed"))
        cfg["tau_inh"] = param_row("tau_inh", "抑制性时间常数 (ms)",
                                   lambda: st.slider("tau_inh", 2.0, 12.0, base_cfg["tau_inh"], 0.5,
                                                     label_visibility="collapsed"))

        st.markdown("#### ⚡ DBS刺激参数")
        cfg["dbs_freq"] = param_row("dbs_freq", "DBS频率 (Hz)",
                                    lambda: st.slider("dbs_freq", 20, 180, base_cfg["dbs_freq"], 5,
                                                      label_visibility="collapsed"))
        cfg["dbs_amp"] = param_row("dbs_amp", "DBS刺激电流",
                                   lambda: st.slider("dbs_amp", 10, 120, base_cfg["dbs_amp"], 5,
                                                     label_visibility="collapsed"))
        cfg["dbs_pulse_width"] = param_row("dbs_pulse_width", "DBS脉宽 (ms)",
                                           lambda: st.number_input("dbs_pulse_width", value=base_cfg["dbs_pulse_width"],
                                                                   min_value=0.05, max_value=1.0, step=0.05,
                                                                   label_visibility="collapsed"))
        cfg["dbs_start_ms"] = param_row("dbs_start_ms", "DBS开启时刻 (ms)",
                                        lambda: st.slider("dbs_start_ms", 200, 2000, int(base_cfg["dbs_start_ms"]), 100,
                                                          label_visibility="collapsed"))

    run_btn = st.button("▶ 运行三工况仿真（正常 / PD / PD+DBS）")
    if run_btn:
        with st.spinner("正在仿真计算，请等待，神经元数量越大耗时越长……"):
            res_norm, res_pd, res_pddbs, fs = run_all_three_conditions(cfg)

            tab1, tab2, tab3 = st.tabs(["🟢 正常", "🔴 PD 患病", "🔵 PD + DBS"])
            for tab, res, tag in [(tab1, res_norm, "正常"), (tab2, res_pd, "PD患病"), (tab3, res_pddbs, "PD+DBS刺激")]:
                with tab:
                    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                    plot_raster(axes[0], res["t"], res["spike"]["GPi"], title=f"{tag} GPi 放电栅格图")
                    plot_lfp(axes[1], res["t"], res["lfp"]["GPi"], title=f"{tag} GPi LFP")
                    plot_stft(axes[2], res["t"], res["lfp"]["GPi"], fs=fs, title=f"{tag} GPi 时频图")
                    fig.tight_layout()
                    st.pyplot(fig)
            st.success("✅ 仿真完成！三个 Tab 分别对应：正常 / PD 患病 / PD+DBS 刺激。")

# ---------------------- 模式2：CSV上传 ----------------------
else:
    upload_file = st.file_uploader("上传csv仿真数据文件", type=["csv"])
    if upload_file is not None:
        df = pd.read_csv(upload_file)
    else:
        st.info("⬆️ 请上传CSV格式仿真结果文件")

if df is not None:
    st.subheader("🎚️ 波形缩放调节")
    col_reset, col_x, col_y = st.columns([1, 3, 3])
    with col_reset:
        reset_btn = st.button("🔄 恢复全局视图")
    with col_x:
        x_min, x_max = st.slider("⏱️ 时间范围（ms）", min_value=float(df["time(ms)"].min()),
                                 max_value=float(df["time(ms)"].max()),
                                 value=(float(df["time(ms)"].min()), float(df["time(ms)"].max())), step=1.0)
    with col_y:
        all_y = df[["STN", "GPe", "GPi"]]
        y_min_global = float(all_y.min().min());
        y_max_global = float(all_y.max().max())
        y_min, y_max = st.slider("⚡ 膜电位范围（mV）", min_value=y_min_global, max_value=y_max_global,
                                 value=(y_min_global, y_max_global), step=0.01)
    st.sidebar.header("📈 绘图设置")
    selected_cols = st.sidebar.multiselect("选择需要绘制的核团信号", options=["STN", "GPe", "GPi"],
                                           default=["STN", "GPe", "GPi"])
    if selected_cols:
        fig = px.line(df, x="time(ms)", y=selected_cols, title="核团膜电位时序曲线",
                      labels={"value": "膜电位 (mV)", "time(ms)": "时间 (ms)", "variable": "核团"})
        fig.update_layout(height=620, xaxis_range=[x_min, x_max], yaxis_range=[y_min, y_max])
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": True,
                                                               "modeBarButtonsToRemove": ["select2d", "lasso2d",
                                                                                          "zoom2d"]})
        st.info("💡 使用方法：拖动时间滑块截取一小段时间，就实现放大；点【恢复全局视图】回到全貌！")
    else:
        st.warning("请至少勾选一个核团信号进行绘图")

# ====================== 休闲窗口：2048 ======================
with st.expander("🎮 休闲窗口 | 2048", expanded=False):
    st.markdown("点击方向按钮或使用键盘方向键控制方块移动，合成 2048 即胜利！")
    game_html_2048 = """
<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
*{box-sizing:border-box}body{margin:0;padding:5px}.game-container{width:340px;margin:10px auto;font-family:Arial}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;background:#bbada0;padding:8px;border-radius:6px}
.cell{width:70px;height:70px;background:#cdc1b4;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:bold;color:#776e65}
.t2{background:#eee4da}.t4{background:#ede0c8}.t8{background:#f2b179;color:#fff}.t16{background:#f59563;color:#fff}
.t32{background:#f67c5f;color:#fff}.t64{background:#f65e3b;color:#fff}.t128{background:#edcf72;color:#fff;font-size:20px}
.t256{background:#edcc61;color:#fff;font-size:20px}.t512{background:#edc850;color:#fff;font-size:20px}.t1024{background:#edc53f;color:#fff;font-size:18px}.t2048{background:#edc22e;color:#fff;font-size:18px}
.control{margin-top:15px;text-align:center}.btn{width:70px;height:50px;margin:4px;font-size:20px;border:none;border-radius:8px;background:#d9d2c6;color:#444;font-weight:bold;cursor:pointer}.btn:active{background:#b9b09f}
</style></head><body>
<div class="game-container"><div class="grid" id="grid"></div><p id="tip" style="text-align:center;font-size:20px">分数：0</p>
<div class="control"><div><button class="btn" onclick="move('up')">↑</button></div><div><button class="btn" onclick="move('left')">←</button><button class="btn" onclick="move('down')">↓</button><button class="btn" onclick="move('right')">→</button></div></div></div>
<script>
const gridEl=document.getElementById("grid"),tipEl=document.getElementById("tip");let grid=Array(4).fill().map(()=>Array(4).fill(0)),score=0;
function randEmpty(){let arr=[];for(let i=0;i<4;i++)for(let j=0;j<4;j++)if(grid[i][j]===0)arr.push([i,j]);if(!arr.length)return null;return arr[Math.floor(Math.random()*arr.length)];}
function addBlock(){let p=randEmpty();if(p)grid[p[0]][p[1]]=Math.random()>0.1?2:4;}
function render(){gridEl.innerHTML="";for(let i=0;i<4;i++)for(let j=0;j<4;j++){let v=grid[i][j],d=document.createElement("div");d.className="cell "+(v?"t"+v:"");d.innerText=v||"";gridEl.appendChild(d);}tipEl.innerText="分数："+score;}
function slide(row){let a=row.filter(x=>x!==0);for(let i=0;i<a.length-1;i++)if(a[i]==a[i+1]){a[i]*=2;score+=a[i];a.splice(i+1,1);}while(a.length<4)a.push(0);return a;}
function move(dir){let old=JSON.stringify(grid);if(dir=="left")for(let i=0;i<4;i++)grid[i]=slide(grid[i]);if(dir=="right")for(let i=0;i<4;i++)grid[i]=slide(grid[i].reverse()).reverse();if(dir=="up")for(let j=0;j<4;j++){let c=[grid[0][j],grid[1][j],grid[2][j],grid[3][j]];c=slide(c);for(let i=0;i<4;i++)grid[i][j]=c[i];}if(dir=="down")for(let j=0;j<4;j++){let c=[grid[0][j],grid[1][j],grid[2][j],grid[3][j]];c=slide(c.reverse()).reverse();for(let i=0;i<4;i++)grid[i][j]=c[i];}if(JSON.stringify(grid)!==old)addBlock();render();}
document.onkeydown=e=>{if(e.key.includes("Arrow")){e.preventDefault();move(e.key.replace("Arrow","").toLowerCase());}}
addBlock();addBlock();render();
</script></body></html>
"""
    html(game_html_2048, height=580)

# ====================== 休闲窗口：3D 飞机大战（Boss战版） ======================
with st.expander("✈️ 休闲窗口 | 极简 3D 飞机大战", expanded=False):
    st.markdown("使用键盘 **方向键/WASD** 或 **滑动屏幕** 移动。达到 200 分会引出 Boss，注意躲避红色子弹！")
    game_html_plane = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body { margin: 0; padding: 0; overflow: hidden; background-color: #050510; font-family: -apple-system, sans-serif; }
#game-container { width: 100%; height: 560px; position: relative; touch-action: none; }
#ui-layer {
    position: absolute; top: 20px; left: 20px; right: 20px;
    display: flex; justify-content: space-between; align-items: flex-start; pointer-events: none; z-index: 5;
}
.score-box {
    background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px);
    padding: 8px 20px; border-radius: 999px; font-size: 16px; font-weight: bold; color: #fff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.2);
    display: flex; gap: 15px; align-items: center;
}
.hearts { color: #ff3b30; font-size: 20px; letter-spacing: 2px; }

#boss-hp-container {
    position: absolute; top: 60px; left: 50%; transform: translateX(-50%);
    width: 70%; max-width: 600px; display: none; flex-direction: column; align-items: center;
    pointer-events: none; z-index: 5;
}
.boss-name { color: #ff3b30; font-weight: bold; font-size: 18px; margin-bottom: 8px; text-shadow: 0 0 10px rgba(255,59,48,0.8); }
.hp-bar-bg {
    width: 100%; height: 16px; background: rgba(255,255,255,0.2); border-radius: 10px; overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.3);
}
.hp-bar-fill { width: 100%; height: 100%; background: linear-gradient(90deg, #ff3b30, #ff9500); transition: width 0.1s ease-out; }

#alert-box {
    position: absolute; top: 120px; left: 50%; transform: translateX(-50%);
    background: rgba(255, 149, 0, 0.9); color: #fff; padding: 10px 25px; border-radius: 999px;
    font-size: 18px; font-weight: bold; display: none; backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(255,149,0,0.5); animation: pulse 1s infinite alternate;
    z-index: 5;
}
@keyframes pulse { from { transform: translateX(-50%) scale(1); } to { transform: translateX(-50%) scale(1.05); } }

#ctrl-layer {
    position: absolute; bottom: 20px; left: 0; right: 0;
    display: flex; justify-content: space-between; padding: 0 20px; pointer-events: none; z-index: 5;
}
.dpad { display: grid; grid-template-columns: repeat(3, 45px); grid-template-rows: repeat(2, 45px); gap: 8px; pointer-events: auto; }
.action-pad { display: flex; flex-direction: column; gap: 15px; justify-content: center; pointer-events: auto; }
.ctrl-btn {
    cursor: pointer; width: 45px; height: 45px; border-radius: 12px;
    background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; color: #fff; border: 1px solid rgba(255,255,255,0.3); user-select: none;
}
.ctrl-btn:active { transform: scale(0.9); background: rgba(255, 255, 255, 0.4); }
.btn-reset { width: 70px; height: 36px; font-size: 14px; border-radius: 18px; background: #0071e3; border: none; color: white; }
#btn-up { grid-column: 2; grid-row: 1; }
#btn-left { grid-column: 1; grid-row: 2; }
#btn-down { grid-column: 2; grid-row: 2; }
#btn-right { grid-column: 3; grid-row: 2; }

/* 增加静音按钮 */
#mute-btn {
    pointer-events: auto; cursor: pointer; position: absolute; bottom: 20px; right: 20px;
    width: 40px; height: 40px; border-radius: 50%; background: rgba(255,255,255,0.15);
    display: flex; align-items: center; justify-content: center; color: white; border: 1px solid rgba(255,255,255,0.3); z-index: 6;
}

#game-over-screen {
    position: absolute; top: 0; left: 0; width: 100%; height: 70%;
    background: rgba(0,0,0,0.85); backdrop-filter: blur(8px);
    display: none; flex-direction: column; align-items: center; justify-content: center;
    color: white; pointer-events: none; z-index: 10;
    border-bottom-left-radius: 20px; border-bottom-right-radius: 20px;
}
#game-over-screen h2 { font-size: 40px; margin-bottom: 10px; color: #ff3b30; }
#btn-restart-center {
    pointer-events: auto; cursor: pointer; margin-top: 20px;
    padding: 12px 30px; border-radius: 999px; font-size: 18px; font-weight: bold;
    background: #0071e3; color: white; border: none; box-shadow: 0 4px 15px rgba(0,113,227,0.5);
    transition: all 0.2s;
}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<div id="game-container">
    <div id="ui-layer">
        <div class="score-box">
            <div>分数: <span id="score">0</span></div>
            <div>关卡: <span id="level">1</span></div>
            <div class="hearts" id="hearts">❤️❤️❤️</div>
        </div>
    </div>

    <div id="boss-hp-container">
        <div class="boss-name">⚠️ 敌方母舰</div>
        <div class="hp-bar-bg"><div class="hp-bar-fill" id="boss-hp-bar"></div></div>
    </div>

    <div id="alert-box">⚠️ 距离 Boss 出现还有 <span id="alert-score">120</span> 分</div>
    <div id="mute-btn">🔊</div>

    <div id="game-over-screen">
        <h2>Game Over</h2>
        <p>最终得分: <span id="final-score">0</span></p>
        <button id="btn-restart-center">🔄 重新起飞</button>
    </div>

    <div id="ctrl-layer">
        <div class="dpad">
            <button class="ctrl-btn" id="btn-up">⬆️</button>
            <button class="ctrl-btn" id="btn-left">⬅️</button>
            <button class="ctrl-btn" id="btn-down">⬇️</button>
            <button class="ctrl-btn" id="btn-right">➡️</button>
        </div>
        <div class="action-pad">
            <button class="ctrl-btn btn-reset" id="btn-reset">🔄 重置</button>
        </div>
    </div>
</div>
<script>
// ================= 音频引擎 =================
const AudioContext = window.AudioContext || window.webkitAudioContext;
let audioCtx = null;
let isMuted = false;
let bgmInterval = null;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new AudioContext();
        startBGM();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
}

// 播放单音
function playTone(freq, type, duration, vol=0.1) {
    if (!audioCtx || isMuted) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(vol, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
}

// 射击音效
function soundShoot() { playTone(800, 'square', 0.1, 0.05); }
// 爆炸音效
function soundExplosion() { playTone(150, 'sawtooth', 0.3, 0.1); }
// 受击音效
function soundHit() { playTone(100, 'square', 0.2, 0.1); }

// 简单的 8-bit BGM 循环
const bgmNotes = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63, 261.63, 196.00];
let bgmIndex = 0;
function startBGM() {
    if (bgmInterval) clearInterval(bgmInterval);
    bgmInterval = setInterval(() => {
        if (!audioCtx || isMuted) return;
        playTone(bgmNotes[bgmIndex], 'triangle', 0.2, 0.03);
        bgmIndex = (bgmIndex + 1) % bgmNotes.length;
    }, 250);
}

document.getElementById('mute-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    initAudio();
    isMuted = !isMuted;
    e.target.innerText = isMuted ? '🔇' : '🔊';
});
// ============================================

const container = document.getElementById('game-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);
scene.fog = new THREE.FogExp2(0x050510, 0.002);

const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
camera.position.set(0, 25, 40); camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dirLight = new THREE.DirectionalLight(0xffffff, 1);
dirLight.position.set(0, 50, 20); scene.add(dirLight);

const starGeo = new THREE.BufferGeometry();
const starPos = new Float32Array(600 * 3);
for(let i=0; i<600*3; i+=3) {
    starPos[i] = (Math.random() - 0.5) * 300;
    starPos[i+1] = (Math.random() - 0.5) * 100;
    starPos[i+2] = -Math.random() * 200;
}
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.5 })));

const playerGroup = new THREE.Group();
const pBody = new THREE.Mesh(new THREE.ConeGeometry(1.5, 5, 4), new THREE.MeshPhongMaterial({ color: 0xffffff }));
pBody.rotation.x = -Math.PI / 2; pBody.rotation.z = Math.PI / 4; playerGroup.add(pBody);
playerGroup.add(new THREE.Mesh(new THREE.BoxGeometry(6, 0.2, 1.5), new THREE.MeshPhongMaterial({ color: 0x0071e3 })));
const engine = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 1), new THREE.MeshBasicMaterial({ color: 0x00e5ff }));
engine.rotation.x = Math.PI / 2; engine.position.set(0, 0, 2); playerGroup.add(engine);
playerGroup.position.set(0, 5, 15); scene.add(playerGroup);

let score = 0, level = 1, hp = 3;
let isGameOver = false, isBossActive = false, isAlertActive = false;
let playerX = 0, playerY = 5;
let boss = null, bossMaxHp = 100, bossCurrentHp = 100;

const scoreEl = document.getElementById('score');
const levelEl = document.getElementById('level');
const heartsEl = document.getElementById('hearts');
const alertBox = document.getElementById('alert-box');
const alertScoreEl = document.getElementById('alert-score');
const bossHpContainer = document.getElementById('boss-hp-container');
const bossHpBar = document.getElementById('boss-hp-bar');
const gameOverScreen = document.getElementById('game-over-screen');
const finalScoreEl = document.getElementById('final-score');

const bullets = [], enemies = [], bossBullets = [];
const bulletGeo = new THREE.SphereGeometry(0.3, 8, 8);
const bulletMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
const enemyGeo = new THREE.ConeGeometry(1.5, 3, 4);
const enemyMat = new THREE.MeshPhongMaterial({ color: 0xff3b30 });
const bossBulletGeo = new THREE.SphereGeometry(0.6, 8, 8);
const bossBulletMat = new THREE.MeshBasicMaterial({ color: 0xff3b30 });

const keys = { ArrowUp: false, ArrowDown: false, ArrowLeft: false, ArrowRight: false, w: false, a: false, s: false, d: false };
window.addEventListener('keydown', (e) => { if (keys.hasOwnProperty(e.key)) keys[e.key] = true; initAudio(); });
window.addEventListener('keyup', (e) => { if (keys.hasOwnProperty(e.key)) keys[e.key] = false; });
function bindBtn(id, key) {
    const btn = document.getElementById(id);
    btn.addEventListener('touchstart', (e) => { e.preventDefault(); keys[key] = true; initAudio(); });
    btn.addEventListener('touchend', (e) => { e.preventDefault(); keys[key] = false; });
    btn.addEventListener('mousedown', () => { keys[key] = true; initAudio(); });
    btn.addEventListener('mouseup', () => keys[key] = false);
    btn.addEventListener('mouseleave', () => keys[key] = false);
}
bindBtn('btn-up', 'ArrowUp'); bindBtn('btn-down', 'ArrowDown'); bindBtn('btn-left', 'ArrowLeft'); bindBtn('btn-right', 'ArrowRight');

function shoot() {
    if (isGameOver) return;
    const b = new THREE.Mesh(bulletGeo, bulletMat);
    b.position.set(playerGroup.position.x, playerGroup.position.y, playerGroup.position.z - 3);
    scene.add(b); bullets.push(b);
    soundShoot();
}

function spawnEnemy() {
    if (isGameOver || isBossActive) return;
    const e = new THREE.Mesh(enemyGeo, enemyMat);
    e.rotation.x = Math.PI / 2; e.rotation.z = Math.PI / 4;
    e.position.set((Math.random() - 0.5) * 30, 5, -100);
    scene.add(e); enemies.push(e);
}

function takeDamage() {
    if (isGameOver) return;
    hp--;
    soundHit();
    heartsEl.innerText = '❤️'.repeat(Math.max(0, hp)) + '🖤'.repeat(3 - Math.max(0, hp));
    document.body.style.backgroundColor = '#ff3b30';
    setTimeout(() => document.body.style.backgroundColor = '#050510', 100);
    if (hp <= 0) {
        isGameOver = true;
        finalScoreEl.innerText = score;
        gameOverScreen.style.display = 'flex';
    }
}

function startBoss() {
    isBossActive = true; isAlertActive = false;
    alertBox.style.display = 'none';
    bossHpContainer.style.display = 'flex';

    bossMaxHp = level * 100; bossCurrentHp = bossMaxHp;
    bossHpBar.style.width = '100%';

    boss = new THREE.Group();
    const bBody = new THREE.Mesh(new THREE.OctahedronGeometry(6), new THREE.MeshPhongMaterial({ color: 0xff3b30, emissive: 0x550000 }));
    boss.add(bBody);
    const bRing = new THREE.Mesh(new THREE.TorusGeometry(7, 0.5, 16, 100), new THREE.MeshBasicMaterial({ color: 0xff9500 }));
    bRing.rotation.x = Math.PI / 2; boss.add(bRing);

    boss.position.set(0, 8, -40); scene.add(boss);
}

function resetGame() {
    score = 0; level = 1; hp = 3; isGameOver = false; isBossActive = false; isAlertActive = false;
    scoreEl.innerText = '0'; levelEl.innerText = '1'; heartsEl.innerText = '❤️❤️❤️';
    playerGroup.position.set(0, 5, 15); playerX = 0; playerY = 5;

    alertBox.style.display = 'none';
    bossHpContainer.style.display = 'none';
    gameOverScreen.style.display = 'none';

    bullets.forEach(b => scene.remove(b)); bullets.length = 0;
    enemies.forEach(e => scene.remove(e)); enemies.length = 0;
    bossBullets.forEach(b => scene.remove(b)); bossBullets.length = 0;
    if (boss) { scene.remove(boss); boss = null; }
}

document.getElementById('btn-reset').addEventListener('click', resetGame);
document.getElementById('btn-restart-center').addEventListener('click', resetGame);

let lastShootTime = 0;
let lastBossShootTime = 0;

function animate() {
    requestAnimationFrame(animate);

    if (!isGameOver) {
        const speed = 0.6;
        if (keys.ArrowUp || keys.w) playerY += speed;
        if (keys.ArrowDown || keys.s) playerY -= speed;
        if (keys.ArrowLeft || keys.a) playerX -= speed;
        if (keys.ArrowRight || keys.d) playerX += speed;
        playerX = Math.max(-15, Math.min(15, playerX));
        playerY = Math.max(2, Math.min(10, playerY));
        playerGroup.position.x += (playerX - playerGroup.position.x) * 0.2;
        playerGroup.position.y += (playerY - playerGroup.position.y) * 0.2;

        const now = Date.now();
        if (now - lastShootTime > 150) { shoot(); lastShootTime = now; }

        if (!isBossActive && !isAlertActive && score >= 80) {
            isAlertActive = true; alertBox.style.display = 'block';
        }
        if (isAlertActive && !isBossActive) {
            let diff = 200 - score;
            if (diff > 0) alertScoreEl.innerText = Math.floor(diff);
            if (score >= 200) startBoss();
        }

        for (let i = bullets.length - 1; i >= 0; i--) {
            let b = bullets[i];
            b.position.z -= 1.5;
            if (b.position.z < -100) { scene.remove(b); bullets.splice(i, 1); continue; }

            let hit = false;
            for (let j = enemies.length - 1; j >= 0; j--) {
                let e = enemies[j];
                if (Math.abs(b.position.x - e.position.x) < 2 && Math.abs(b.position.y - e.position.y) < 2 && Math.abs(b.position.z - e.position.z) < 3) {
                    scene.remove(b); bullets.splice(i, 1); scene.remove(e); enemies.splice(j, 1);
                    soundExplosion();
                    if(!isBossActive) { score += 10; scoreEl.innerText = score; }
                    hit = true; break;
                }
            }
            if (!hit && isBossActive && boss) {
                if (Math.abs(b.position.x - boss.position.x) < 6 && Math.abs(b.position.z - boss.position.z) < 6) {
                    scene.remove(b); bullets.splice(i, 1);
                    bossCurrentHp--;
                    bossHpBar.style.width = `${(bossCurrentHp / bossMaxHp) * 100}%`;
                    if (bossCurrentHp <= 0) {
                        scene.remove(boss); boss = null;
                        isBossActive = false;
                        bossHpContainer.style.display = 'none';
                        level++; levelEl.innerText = level;
                        score = 0; scoreEl.innerText = '0';
                        soundExplosion();
                    }
                }
            }
        }

        for (let i = enemies.length - 1; i >= 0; i--) {
            let e = enemies[i];
            e.position.z += 0.4;
            if (Math.abs(e.position.x - playerGroup.position.x) < 2.5 && Math.abs(e.position.z - playerGroup.position.z) < 3) {
                scene.remove(e); enemies.splice(i, 1);
                takeDamage();
            }
            if (e.position.z > 20) { scene.remove(e); enemies.splice(i, 1); }
        }

        if (!isBossActive && Math.random() < 0.03) spawnEnemy();

        if (isBossActive && boss) {
            if (boss.position.z < -15) boss.position.z += 0.1;
            boss.position.x = Math.sin(now * 0.001) * 10;

            if (now - lastBossShootTime > 600) {
                const bb = new THREE.Mesh(bossBulletGeo, bossBulletMat);
                bb.position.copy(boss.position);
                scene.add(bb); bossBullets.push(bb);
                lastBossShootTime = now;
            }
        }

        for (let i = bossBullets.length - 1; i >= 0; i--) {
            let bb = bossBullets[i];
            if (bb.position.x < playerGroup.position.x) bb.position.x += 0.05;
            if (bb.position.x > playerGroup.position.x) bb.position.x -= 0.05;
            bb.position.z += 0.5;

            if (Math.abs(bb.position.x - playerGroup.position.x) < 2 && Math.abs(bb.position.z - playerGroup.position.z) < 2) {
                scene.remove(bb); bossBullets.splice(i, 1);
                takeDamage();
            } else if (bb.position.z > 20) {
                scene.remove(bb); bossBullets.splice(i, 1);
            }
        }
    }

    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
});

resetGame();
animate();
</script>
</body>
</html>
"""
    html(game_html_plane, height=560)
# ====================== 休闲窗口：黄金矿工 ======================
with st.expander("🎮 休闲窗口 | 黄金矿工", expanded=False):
    st.markdown("按 **空格键** 或点击 **放钩** 按钮抓取金矿、钻石。限时 60 秒！")
    game_html_gold = """
<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
*{box-sizing:border-box}body{margin:0;padding:8px;font-family:-apple-system,BlinkMacSystemFont,Arial;background:#fdfaf6}
.box{width:600px;margin:0 auto;text-align:center}
canvas{background:#f4ebd9;border-radius:12px;display:block;margin:0 auto;box-shadow:0 8px 24px rgba(0,0,0,0.08);border:1px solid #e0d4c0}
.ctrl{margin-top:16px}
button{width:120px;height:46px;font-size:16px;border:none;border-radius:10px;background:#007aff;color:#fff;font-weight:600;cursor:pointer;box-shadow:0 4px 12px rgba(0,122,255,0.25);transition:0.2s}
button:active{transform:translateY(2px);box-shadow:none}
button:hover{background:#0084ff}
</style></head><body>
<div class="box">
<canvas id="gameCanvas" width="600" height="360"></canvas>
<div class="ctrl"><button id="dropBtn">放钩 (Space)</button></div>
</div>
<script>
const canvas=document.getElementById('gameCanvas'),ctx=canvas.getContext('2d');
const hookOrigin={x:300,y:60},hook={length:30,minLen:30,maxLen:450,angle:0,angleSpeed:0.025,state:'swing',speed:6,target:null};
let score=0,timeLeft=60,gameActive=true,timerId=null,minerals=[];
// 矿物类型
const types=[
    {name:'大金块',value:100,r:16,color:'#ffd700'},
    {name:'小黄金',value:50,r:10,color:'#ffcc00'},
    {name:'钻石',value:300,r:9,color:'#00e5ff'},
    {name:'石头',value:20,r:20,color:'#888'}
];
function initMinerals(){
    minerals=[];
    for(let i=0;i<14;i++){
        let type=types[Math.floor(Math.random()*types.length)];
        // 避免生成在太靠边缘和顶部的区域
        let x=50+Math.random()*500;
        let y=140+Math.random()*190;
        minerals.push({x,y,type});
    }
}
function startGame(){
    score=0;timeLeft=60;gameActive=true;hook.state='swing';hook.angle=0;hook.length=30;hook.target=null;
    initMinerals();
    if(timerId)clearInterval(timerId);
    timerId=setInterval(()=>{
        if(!gameActive)return;
        timeLeft--;
        if(timeLeft<=0){timeLeft=0;gameActive=false;}
    },1000);
}
function dropHook(){if(gameActive&&hook.state==='swing')hook.state='drop';}
function checkCollision(){
    let tipX=hookOrigin.x+Math.sin(hook.angle)*hook.length;
    let tipY=hookOrigin.y+Math.cos(hook.angle)*hook.length;
    for(let i=0;i<minerals.length;i++){
        let m=minerals[i];
        if(Math.hypot(tipX-m.x,tipY-m.y)<m.type.r+12){
            hook.state='pull';hook.target=m;minerals.splice(i,1);return;
        }
    }
}
function update(){
    if(!gameActive)return;
    if(hook.state==='swing'){
        hook.angle+=hook.angleSpeed;
        if(hook.angle>1.3||hook.angle<-1.3)hook.angleSpeed*=-1;
    }else if(hook.state==='drop'){
        hook.length+=hook.speed;
        checkCollision();
        if(hook.length>hook.maxLen)hook.state='pull';
    }else if(hook.state==='pull'){
        hook.length-=hook.speed;
        if(hook.length<=hook.minLen){
            if(hook.target){score+=hook.target.type.value;hook.target=null;}
            hook.state='swing';
        }
    }
}
function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);

    // 顶部信息栏背景
    ctx.fillStyle='rgba(0,0,0,0.05)';
    ctx.fillRect(0,0,canvas.width,50);

    // 文字信息
    ctx.fillStyle='#333';ctx.font='bold 20px Arial';ctx.textAlign='left';
    ctx.fillText(`分数: ${score}`,30,33);
    ctx.textAlign='right';
    ctx.fillText(`时间: ${timeLeft}s`,canvas.width-30,33);

    // 画矿物
    minerals.forEach(m=>{
        ctx.beginPath();ctx.arc(m.x,m.y,m.type.r,0,Math.PI*2);
        ctx.fillStyle=m.type.color;ctx.fill();
        ctx.strokeStyle='#333';ctx.lineWidth=1.5;ctx.stroke();
    });

    // 钩子和绳子
    let tipX=hookOrigin.x+Math.sin(hook.angle)*hook.length;
    let tipY=hookOrigin.y+Math.cos(hook.angle)*hook.length;
    ctx.beginPath();ctx.moveTo(hookOrigin.x,hookOrigin.y);ctx.lineTo(tipX,tipY);
    ctx.strokeStyle='#555';ctx.lineWidth=3;ctx.stroke();

    // 抓到的矿物
    if(hook.target){
        ctx.beginPath();ctx.arc(tipX,tipY+hook.target.type.r,hook.target.type.r,0,Math.PI*2);
        ctx.fillStyle=hook.target.type.color;ctx.fill();
        ctx.strokeStyle='#333';ctx.lineWidth=1.5;ctx.stroke();
    }

    // 钩子
    ctx.beginPath();ctx.arc(tipX,tipY,7,0,Math.PI*2);
    ctx.fillStyle='#666';ctx.fill();

    // 矿工（可爱的圆脑袋）
    ctx.beginPath();ctx.arc(hookOrigin.x,hookOrigin.y-15,18,0,Math.PI*2);
    ctx.fillStyle='#f4a460';ctx.fill();
    ctx.strokeStyle='#c77e3a';ctx.lineWidth=2;ctx.stroke();

    if(!gameActive){
        ctx.fillStyle='rgba(0,0,0,0.65)';ctx.fillRect(0,0,canvas.width,canvas.height);
        ctx.fillStyle='#fff';ctx.font='bold 32px Arial';ctx.textAlign='center';
        ctx.fillText('时间到！最终分数: '+score,canvas.width/2,canvas.height/2-10);
        ctx.font='18px Arial';ctx.fillStyle='#ddd';
        ctx.fillText('点击下方按钮重新开始',canvas.width/2,canvas.height/2+30);
    }
}
function loop(){update();draw();requestAnimationFrame(loop);}
document.getElementById('dropBtn').addEventListener('click',()=>{if(!gameActive)startGame();else dropHook();});
document.addEventListener('keydown',(e)=>{if(e.code==='Space'){e.preventDefault();if(!gameActive)startGame();else dropHook();}});
startGame();loop();
</script></body></html>
"""
    html(game_html_gold, height=460)

# ====================== 休闲窗口：3D 赛车 ======================
with st.expander("🏎️ 休闲窗口 | 极简 3D 赛车", expanded=False):
    st.markdown("使用键盘 **←** / **→** 变道，或点击下方按钮。躲避红色路障，速度会越来越快！")
    game_html_racing = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body { margin: 0; padding: 0; overflow: hidden; background-color: #fbfbfd; font-family: -apple-system, sans-serif; }
#game-container { width: 100%; height: 520px; position: relative; touch-action: none; }
#ui-layer {
    position: absolute; top: 20px; left: 20px; right: 20px;
    display: flex; justify-content: space-between; pointer-events: none;
}
.score-box {
    background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(10px);
    padding: 8px 20px; border-radius: 999px; font-size: 18px; font-weight: bold; color: #1d1d1f;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
/* 底部苹果风控制面板 */
#ctrl-layer {
    position: absolute; bottom: 20px; left: 0; right: 0;
    display: flex; justify-content: center; gap: 20px; pointer-events: none;
}
.ctrl-btn {
    pointer-events: auto; cursor: pointer;
    width: 70px; height: 70px; border-radius: 50%;
    background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; color: #1d1d1f; border: none; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: all 0.2s; user-select: none;
}
.ctrl-btn:active { transform: scale(0.9); background: #e8e8ed; }
.ctrl-btn.reset { width: 60px; height: 60px; font-size: 20px; margin-top: 5px; }
</style>
<!-- 引入 Three.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<div id="game-container">
    <div id="ui-layer">
        <div class="score-box">分数: <span id="score">0</span></div>
        <div class="score-box">最高分: <span id="highscore">0</span></div>
    </div>
    <div id="ctrl-layer">
        <button class="ctrl-btn" id="btn-left">⬅️</button>
        <button class="ctrl-btn reset" id="btn-reset">🔄</button>
        <button class="ctrl-btn" id="btn-right">➡️</button>
    </div>
</div>
<script>
// 1. 初始化场景
const container = document.getElementById('game-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xddeeff);
// 雾效范围调整，让远处的路障显现出来
scene.fog = new THREE.Fog(0xddeeff, 20, 80); 

// 调整了摄像机位置和俯仰角，能看到玩家的车屁股
const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
camera.position.set(0, 8, 18);
camera.lookAt(0, 0, -10);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

// 2. 光照
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(10, 20, 10);
dirLight.castShadow = true;
scene.add(dirLight);

// 3. 场景对象
const roadGeo = new THREE.PlaneGeometry(20, 200);
const roadMat = new THREE.MeshLambertMaterial({ color: 0x555555 });
const road = new THREE.Mesh(roadGeo, roadMat);
road.rotation.x = -Math.PI / 2;
road.position.z = -50;
road.receiveShadow = true;
scene.add(road);

const lineMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
const laneLines = [];
for (let i = -1; i <= 1; i+=2) {
    for (let j = 0; j < 20; j++) {
        const lineGeo = new THREE.PlaneGeometry(0.5, 4);
        const line = new THREE.Mesh(lineGeo, lineMat);
        line.rotation.x = -Math.PI / 2;
        line.position.set(i * 3.33, 0.1, -j * 10);
        scene.add(line);
        laneLines.push(line);
    }
}

// 玩家赛车
const playerCar = new THREE.Group();
const body = new THREE.Mesh(new THREE.BoxGeometry(2, 1, 4), new THREE.MeshPhongMaterial({ color: 0x0071e3, shininess: 100 }));
body.position.y = 0.5; body.castShadow = true; playerCar.add(body);

const roof = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.8, 2), new THREE.MeshPhongMaterial({ color: 0xffffff }));
roof.position.set(0, 1.4, -0.2); roof.castShadow = true; playerCar.add(roof);

// 尾灯
const tailLight = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.2, 0.2), new THREE.MeshBasicMaterial({ color: 0xff3b30 }));
tailLight.position.set(0, 0.7, 2); playerCar.add(tailLight);

playerCar.position.set(0, 0, 0);
scene.add(playerCar);

// 障碍物
const obstacles = [];
const obstacleGeo = new THREE.BoxGeometry(1.8, 1.8, 3);
const obstacleMat = new THREE.MeshPhongMaterial({ color: 0xff3b30 });

// 4. 游戏状态
let score = 0, gameSpeed = 0.4, isGameOver = false, targetLane = 0;
const laneWidth = 3.33;
let highscore = 0;

// 5. 控制逻辑
const scoreEl = document.getElementById('score');
const highscoreEl = document.getElementById('highscore');

function moveLeft() { if (!isGameOver && targetLane > -1) targetLane -= 1; }
function moveRight() { if (!isGameOver && targetLane < 1) targetLane += 1; }

document.getElementById('btn-left').addEventListener('click', moveLeft);
document.getElementById('btn-right').addEventListener('click', moveRight);
document.getElementById('btn-reset').addEventListener('click', resetGame);

document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') moveLeft();
    if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') moveRight();
});

// 触摸滑动控制
let touchStartX = 0;
container.addEventListener('touchstart', e => { touchStartX = e.touches[0].clientX; });
container.addEventListener('touchend', e => {
    let touchEndX = e.changedTouches[0].clientX;
    if (touchEndX < touchStartX - 30) moveLeft();
    if (touchEndX > touchStartX + 30) moveRight();
});

// 6. 障碍物生成与重置
function spawnObstacle() {
    if (isGameOver) return;
    const lane = Math.floor(Math.random() * 3) - 1;
    const obs = new THREE.Mesh(obstacleGeo, obstacleMat);
    obs.position.set(lane * laneWidth, 0.9, -100);
    obs.castShadow = true;
    scene.add(obs);
    obstacles.push(obs);
}

function resetGame() {
    score = 0; gameSpeed = 0.4; isGameOver = false; targetLane = 0;
    scoreEl.innerText = '0';
    camera.position.set(0, 8, 18);

    // 清理障碍物
    obstacles.forEach(obs => scene.remove(obs));
    obstacles.length = 0;
    playerCar.position.x = 0;

    // 清理多余状态
    spawnObstacle();
}

// 7. 动画循环
function animate() {
    requestAnimationFrame(animate);

    if (!isGameOver) {
        playerCar.position.x += (targetLane * laneWidth - playerCar.position.x) * 0.15;

        laneLines.forEach(line => {
            line.position.z += gameSpeed;
            if (line.position.z > 10) line.position.z = -100;
        });

        for (let i = obstacles.length - 1; i >= 0; i--) {
            let obs = obstacles[i];
            obs.position.z += gameSpeed;

            // 碰撞检测
            if (Math.abs(obs.position.z - playerCar.position.z) < 2.5 && 
                Math.abs(obs.position.x - playerCar.position.x) < 1.5) {
                isGameOver = true;
                obs.material.color.setHex(0x000000);
                if (score > highscore) {
                    highscore = score;
                    highscoreEl.innerText = Math.floor(highscore);
                }
            }

            if (obs.position.z > 15) {
                scene.remove(obs);
                obstacles.splice(i, 1);
            }
        }

        if (Math.random() < 0.02 + score * 0.00005) {
            if (obstacles.length === 0 || obstacles[obstacles.length-1].position.z > -30) {
                spawnObstacle();
            }
        }

        score += 0.1;
        scoreEl.innerText = Math.floor(score);
        gameSpeed += 0.0001;
    } else {
        // 碰撞后视角拉远
        camera.position.z += 0.1;
        camera.position.y += 0.05;
        if (camera.position.z > 20) camera.position.z = 20;
    }
    renderer.render(scene, camera);
}

// 自适应窗口
window.addEventListener('resize', () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
});

spawnObstacle();
animate();
</script>
</body>
</html>
"""
    html(game_html_racing, height=560)

# ====================== 休闲窗口：3D 上帝视角坦克大战 ======================
with st.expander("🚀 休闲窗口 | 极简 3D 坦克大战", expanded=False):
    st.markdown("点击下方按钮或使用键盘 **WASD/方向键** 移动，**空格键** 开火。消灭 10 辆敌方坦克即可晋级！")
    game_html_tank = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body { margin: 0; padding: 0; overflow: hidden; background-color: #222; font-family: -apple-system, sans-serif; }
#game-container { width: 100%; height: 560px; position: relative; touch-action: none; }
#ui-layer {
    position: absolute; top: 20px; left: 20px; right: 20px;
    display: flex; justify-content: space-between; align-items: flex-start; pointer-events: none; z-index: 5;
}
.score-box {
    background: rgba(0, 0, 0, 0.6); backdrop-filter: blur(10px);
    padding: 8px 20px; border-radius: 12px; font-size: 16px; font-weight: bold; color: #fff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.2);
    display: flex; flex-direction: column; gap: 5px;
}
.hearts { color: #ff3b30; font-size: 20px; letter-spacing: 2px; }

#ctrl-layer {
    position: absolute; bottom: 20px; left: 0; right: 0;
    display: flex; justify-content: space-between; padding: 0 30px; pointer-events: none; z-index: 5;
}
.dpad { display: grid; grid-template-columns: repeat(3, 50px); grid-template-rows: repeat(3, 50px); gap: 5px; pointer-events: auto; }
.action-pad { display: flex; gap: 15px; align-items: flex-end; pointer-events: auto; }

.ctrl-btn {
    cursor: pointer; width: 50px; height: 50px; border-radius: 50%;
    background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(5px);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: #fff; border: 2px solid rgba(255,255,255,0.4); user-select: none;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3); transition: 0.1s;
}
.ctrl-btn:active { transform: scale(0.9); background: rgba(255, 255, 255, 0.5); }

.btn-fire { width: 80px; height: 80px; font-size: 24px; background: rgba(255, 59, 48, 0.6); border-color: #ff3b30; }
.btn-reset { width: 60px; height: 60px; font-size: 16px; background: rgba(0, 122, 255, 0.6); border-color: #007aff; }

#btn-up { grid-column: 2; grid-row: 1; }
#btn-left { grid-column: 1; grid-row: 2; }
#btn-down { grid-column: 2; grid-row: 3; }
#btn-right { grid-column: 3; grid-row: 2; }

#mute-btn {
    pointer-events: auto; cursor: pointer; position: absolute; bottom: 20px; right: 110px;
    width: 40px; height: 40px; border-radius: 50%; background: rgba(255,255,255,0.15);
    display: flex; align-items: center; justify-content: center; color: white; border: 1px solid rgba(255,255,255,0.3); z-index: 6;
}

#game-over-screen {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.85); backdrop-filter: blur(8px);
    display: none; flex-direction: column; align-items: center; justify-content: center;
    color: white; z-index: 10;
}
#game-over-screen h2 { font-size: 40px; margin-bottom: 10px; color: #ff3b30; }
#btn-restart-center {
    pointer-events: auto; cursor: pointer; margin-top: 20px;
    padding: 12px 30px; border-radius: 999px; font-size: 18px; font-weight: bold;
    background: #007aff; color: white; border: none; box-shadow: 0 4px 15px rgba(0,122,255,0.5);
    transition: all 0.2s;
}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<div id="game-container">
    <div id="ui-layer">
        <div class="score-box">
            <div class="hearts" id="hearts">❤️❤️❤️</div>
            <div>关卡: <span id="level">1</span> (进度: <span id="progress">0</span>/<span id="target">10</span>)</div>
        </div>
        <div class="score-box" style="align-items: flex-end;">
            <div style="font-size: 12px; color:#ccc;">总击杀: <span id="score">0</span></div>
            <div style="font-size: 20px; color:#ffd700;">分数: <span id="total_score">0</span></div>
        </div>
    </div>
    <div id="mute-btn">🔊</div>

    <div id="game-over-screen">
        <h2>坦克被击毁</h2>
        <p>最终得分: <span id="final-score">0</span></p>
        <button id="btn-restart-center">🔄 重新部署</button>
    </div>

    <div id="ctrl-layer">
        <div class="dpad">
            <button class="ctrl-btn" id="btn-up">⬆️</button>
            <button class="ctrl-btn" id="btn-left">⬅️</button>
            <button class="ctrl-btn" id="btn-down">⬇️</button>
            <button class="ctrl-btn" id="btn-right">➡️</button>
        </div>
        <div class="action-pad">
            <button class="ctrl-btn btn-reset" id="btn-reset">🔄</button>
            <button class="ctrl-btn btn-fire" id="btn-fire">💥</button>
        </div>
    </div>
</div>
<script>
// ================= 音频引擎 =================
const AudioContext = window.AudioContext || window.webkitAudioContext;
let audioCtx = null;
let isMuted = false;
let bgmInterval = null;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new AudioContext();
        startBGM();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
}

function playTone(freq, type, duration, vol=0.1) {
    if (!audioCtx || isMuted) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(vol, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
}

// 开火音效（低频轰）
function soundShoot() { playTone(150, 'sawtooth', 0.2, 0.1); }
// 爆炸音效
function soundExplosion() { playTone(80, 'square', 0.4, 0.15); }
// 受击音效
function soundHit() { playTone(60, 'sawtooth', 0.3, 0.15); }

// 8-bit 炮火 BGM
const bgmNotes = [130.81, 164.81, 196.00, 164.81, 130.81, 196.00, 164.81, 130.81];
let bgmIndex = 0;
function startBGM() {
    if (bgmInterval) clearInterval(bgmInterval);
    bgmInterval = setInterval(() => {
        if (!audioCtx || isMuted) return;
        playTone(bgmNotes[bgmIndex], 'square', 0.3, 0.02);
                bgmIndex = (bgmIndex + 1) % bgmNotes.length;
    }, 400);
}

document.getElementById('mute-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    initAudio();
    isMuted = !isMuted;
    e.target.innerText = isMuted ? '🔇' : '🔊';
});
// ============================================

const container = document.getElementById('game-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
scene.fog = new THREE.Fog(0x1a1a1a, 40, 100);

const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
camera.position.set(0, 50, 0);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.6); scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(20, 50, 20); dirLight.castShadow = true;
dirLight.shadow.camera.left = -50; dirLight.shadow.camera.right = 50;
dirLight.shadow.camera.top = 50; dirLight.shadow.camera.bottom = -50;
scene.add(dirLight);

const floorGeo = new THREE.PlaneGeometry(60, 60);
const floorMat = new THREE.MeshLambertMaterial({ color: 0x2a2a2a });
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
const grid = new THREE.GridHelper(60, 30, 0x444444, 0x333333); scene.add(grid);

let hp = 3, level = 1, totalScore = 0, killCount = 0;
let targetKills = 10;
let isGameOver = false;
let player = null;
let enemies = [];
let walls = [];
let bullets = [];
let enemyBullets = [];
let lastShootTime = 0;

const heartsEl = document.getElementById('hearts');
const scoreEl = document.getElementById('score');
const totalScoreEl = document.getElementById('total_score');
const levelEl = document.getElementById('level');
const progressEl = document.getElementById('progress');
const targetEl = document.getElementById('target');
const gameOverScreen = document.getElementById('game-over-screen');
const finalScoreEl = document.getElementById('final-score');

function createTank(color) {
    const group = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(2.5, 0.8, 3), new THREE.MeshPhongMaterial({ color: color }));
    body.position.y = 0.4; body.castShadow = true; body.receiveShadow = true; group.add(body);
    const trackMat = new THREE.MeshPhongMaterial({ color: 0x111111 });
    const trackL = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.6, 3.2), trackMat);
    trackL.position.set(-1.4, 0.3, 0); group.add(trackL);
    const trackR = trackL.clone(); trackR.position.set(1.4, 0.3, 0); group.add(trackR);
    const turret = new THREE.Group(); turret.position.y = 1;
    const tBody = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 1, 0.8, 16), new THREE.MeshPhongMaterial({ color: color }));
    tBody.castShadow = true; turret.add(tBody);
    const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 2.5), new THREE.MeshPhongMaterial({ color: 0x555555 }));
    barrel.rotation.x = Math.PI / 2; barrel.position.set(0, 0, 1.5); barrel.castShadow = true; turret.add(barrel);
    group.add(turret);
    group.userData.turret = turret;
    return group;
}

function generateMap() {
    walls.forEach(w => scene.remove(w)); walls = [];
    const wallMat = new THREE.MeshPhongMaterial({ color: 0x556677 });
    for (let i = 0; i < 25; i++) {
        const w = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 6), wallMat);
        w.rotation.y = Math.random() > 0.5 ? 0 : Math.PI / 2;
        let x, z, valid = false, tries = 0;
        while(!valid && tries < 50) {
            x = (Math.random() - 0.5) * 50;
            z = (Math.random() - 0.5) * 50;
            if (Math.abs(x) < 10 && Math.abs(z) < 10) continue;
            valid = true;
        }
        w.position.set(x, 1, z);
        w.castShadow = true; w.receiveShadow = true;
        scene.add(w);
        walls.push(w);
    }
}

function initPlayer() {
    if (player) scene.remove(player);
    player = createTank(0x007aff);
    player.position.set(0, 0, 0);
    scene.add(player);
}

function spawnEnemy() {
    if (isGameOver || enemies.length >= 4) return;
    const e = createTank(0xff3b30);
    let x, z, dist = 0, tries = 0;
    while (dist < 25 && tries < 100) {
        x = (Math.random() - 0.5) * 50;
        z = (Math.random() - 0.5) * 50;
        dist = Math.hypot(x - player.position.x, z - player.position.z);
        tries++;
    }
    let inWall = false;
    walls.forEach(w => {
        if (Math.abs(x - w.position.x) < 3 && Math.abs(z - w.position.z) < 3) inWall = true;
    });
    if (inWall) { spawnEnemy(); return; }

    e.position.set(x, 0, z);
    e.userData.hp = 1;
    e.userData.speed = 0.05 + level * 0.01;
    e.userData.lastShoot = 0;
    scene.add(e);
    enemies.push(e);
}

function takeDamage() {
    if (isGameOver) return;
    hp--;
    soundHit();
    heartsEl.innerText = '❤️'.repeat(Math.max(0, hp)) + '🖤'.repeat(3 - Math.max(0, hp));
    document.body.style.backgroundColor = '#ff3b30';
    setTimeout(() => document.body.style.backgroundColor = '#222', 100);
    if (hp <= 0) {
        isGameOver = true;
        finalScoreEl.innerText = totalScore;
        gameOverScreen.style.display = 'flex';
    }
}

function resetGame() {
    hp = 3; level = 1; totalScore = 0; killCount = 0;
    isGameOver = false;
    targetKills = 10;
    heartsEl.innerText = '❤️❤️❤️';
    scoreEl.innerText = '0'; totalScoreEl.innerText = '0';
    levelEl.innerText = level; progressEl.innerText = '0'; targetEl.innerText = targetKills;
    gameOverScreen.style.display = 'none';
    bullets.forEach(b => scene.remove(b)); bullets = [];
    enemyBullets.forEach(b => scene.remove(b)); enemyBullets = [];
    enemies.forEach(e => scene.remove(e)); enemies = [];
    generateMap();
    initPlayer();
    while(enemies.length < 4) { spawnEnemy(); }
}

function nextLevel() {
    level++;
    killCount = 0;
    targetKills = level * 10;
    levelEl.innerText = level;
    progressEl.innerText = killCount;
    targetEl.innerText = targetKills;
    generateMap();
    initPlayer();
    enemies.forEach(e => scene.remove(e)); enemies = [];
    while(enemies.length < 4) { spawnEnemy(); }
}

const keys = { ArrowUp: false, ArrowDown: false, ArrowLeft: false, ArrowRight: false, w: false, a: false, s: false, d: false, ' ': false };
window.addEventListener('keydown', (e) => { if (keys.hasOwnProperty(e.key)) { keys[e.key] = true; initAudio(); } });
window.addEventListener('keyup', (e) => { if (keys.hasOwnProperty(e.key)) keys[e.key] = false; });

function bindBtn(id, key) {
    const btn = document.getElementById(id);
    btn.addEventListener('touchstart', (e) => { e.preventDefault(); keys[key] = true; initAudio(); });
    btn.addEventListener('touchend', (e) => { e.preventDefault(); keys[key] = false; });
    btn.addEventListener('mousedown', () => { keys[key] = true; initAudio(); });
    btn.addEventListener('mouseup', () => keys[key] = false);
    btn.addEventListener('mouseleave', () => keys[key] = false);
}
bindBtn('btn-up', 'ArrowUp'); bindBtn('btn-down', 'ArrowDown'); bindBtn('btn-left', 'ArrowLeft'); bindBtn('btn-right', 'ArrowRight'); bindBtn('btn-fire', ' ');
document.getElementById('btn-reset').addEventListener('click', resetGame);
document.getElementById('btn-restart-center').addEventListener('click', resetGame);

function checkWallCollision(newPos, radius) {
    if (Math.abs(newPos.x) > 28 || Math.abs(newPos.z) > 28) return true;
    for (let w of walls) {
        let dx = Math.abs(newPos.x - w.position.x);
        let dz = Math.abs(newPos.z - w.position.z);
        let wx = w.rotation.y === 0 ? 1 : 3;
        let wz = w.rotation.y === 0 ? 3 : 1;
        if (dx < wx + radius && dz < wz + radius) return true;
    }
    return false;
}

let lastPlayerShoot = 0;

function animate() {
    requestAnimationFrame(animate);
    const now = Date.now();

    if (!isGameOver) {
        let moveX = 0, moveZ = 0;
        const speed = 0.25;
        if (keys.ArrowUp || keys.w) moveZ -= speed;
        if (keys.ArrowDown || keys.s) moveZ += speed;
        if (keys.ArrowLeft || keys.a) moveX -= speed;
        if (keys.ArrowRight || keys.d) moveX += speed;

        if (moveX !== 0 || moveZ !== 0) {
            const targetAngle = Math.atan2(moveX, moveZ);
            let diff = targetAngle - player.rotation.y;
            while (diff > Math.PI) diff -= Math.PI * 2;
            while (diff < -Math.PI) diff += Math.PI * 2;
            player.rotation.y += diff * 0.2;
            const nextPos = player.position.clone().add(new THREE.Vector3(moveX, 0, moveZ));
            if (!checkWallCollision(nextPos, 1.5)) {
                player.position.copy(nextPos);
            }
        }

        if (keys[' '] && now - lastPlayerShoot > 1000) {
            const b = new THREE.Mesh(new THREE.SphereGeometry(0.4, 8, 8), new THREE.MeshBasicMaterial({ color: 0x00e5ff }));
            b.position.copy(player.position);
            b.position.x += Math.sin(player.rotation.y) * 2.5;
            b.position.z += Math.cos(player.rotation.y) * 2.5;
            b.position.y = 1;
            b.userData.dx = Math.sin(player.rotation.y);
            b.userData.dz = Math.cos(player.rotation.y);
            b.userData.isPlayer = true;
            scene.add(b);
            bullets.push(b);
            lastPlayerShoot = now;
            soundShoot();
        }

        enemies.forEach(enemy => {
            const angleToPlayer = Math.atan2(player.position.x - enemy.position.x, player.position.z - enemy.position.z);
            enemy.userData.turret.rotation.y += (angleToPlayer - enemy.userData.turret.rotation.y) * 0.1;

            const dir = new THREE.Vector3().subVectors(player.position, enemy.position).normalize();
            const distToPlayer = enemy.position.distanceTo(player.position);
            
            if (distToPlayer > 15) {
                const nextPos = enemy.position.clone().add(dir.multiplyScalar(enemy.userData.speed));
                if (!checkWallCollision(nextPos, 1.5)) {
                    enemy.position.copy(nextPos);
                } else {
                    const leftDir = new THREE.Vector3(dir.z, 0, -dir.x).multiplyScalar(enemy.userData.speed);
                    const rightDir = new THREE.Vector3(-dir.z, 0, dir.x).multiplyScalar(enemy.userData.speed);
                    if (!checkWallCollision(enemy.position.clone().add(leftDir), 1.5)) enemy.position.add(leftDir);
                    else if (!checkWallCollision(enemy.position.clone().add(rightDir), 1.5)) enemy.position.add(rightDir);
                }
            }

            if (now - enemy.userData.lastShoot > 2000 && distToPlayer < 30) {
                const eb = new THREE.Mesh(new THREE.SphereGeometry(0.4, 8, 8), new THREE.MeshBasicMaterial({ color: 0xff3b30 }));
                eb.position.copy(enemy.position);
                eb.position.x += Math.sin(enemy.userData.turret.rotation.y) * 2.5;
                eb.position.z += Math.cos(enemy.userData.turret.rotation.y) * 2.5;
                eb.position.y = 1;
                const shootDir = new THREE.Vector3().subVectors(player.position, enemy.position).normalize();
                eb.userData.dx = shootDir.x;
                eb.userData.dz = shootDir.z;
                eb.userData.isPlayer = false;
                scene.add(eb);
                enemyBullets.push(eb);
                enemy.userData.lastShoot = now;
            }
        });

        const bulletSpeed = 0.35;
        
        for (let i = bullets.length - 1; i >= 0; i--) {
            let b = bullets[i];
            b.position.x += b.userData.dx * bulletSpeed;
            b.position.z += b.userData.dz * bulletSpeed;
            if (checkWallCollision(b.position, 0.2) || Math.abs(b.position.x) > 30 || Math.abs(b.position.z) > 30) {
                scene.remove(b); bullets.splice(i, 1); continue;
            }
            let hit = false;
            for (let j = enemies.length - 1; j >= 0; j--) {
                let e = enemies[j];
                if (b.position.distanceTo(e.position) < 2) {
                    scene.remove(b); bullets.splice(i, 1);
                    scene.remove(e); enemies.splice(j, 1);
                    totalScore += 100; totalScoreEl.innerText = totalScore;
                    killCount++; progressEl.innerText = killCount;
                    scoreEl.innerText = killCount;
                    soundExplosion();
                    setTimeout(spawnEnemy, 500); 
                    hit = true; break;
                }
            }
        }

        for (let i = enemyBullets.length - 1; i >= 0; i--) {
            let b = enemyBullets[i];
            b.position.x += b.userData.dx * bulletSpeed * 0.8;
            b.position.z += b.userData.dz * bulletSpeed * 0.8;
            if (checkWallCollision(b.position, 0.2) || Math.abs(b.position.x) > 30 || Math.abs(b.position.z) > 30) {
                scene.remove(b); enemyBullets.splice(i, 1); continue;
            }
            if (b.position.distanceTo(player.position) < 2) {
                scene.remove(b); enemyBullets.splice(i, 1);
                takeDamage();
            }
        }

        enemies.forEach(e => {
            if (e.position.distanceTo(player.position) < 2.5) {
                takeDamage();
                const bounceDir = new THREE.Vector3().subVectors(e.position, player.position).normalize();
                e.position.add(bounceDir.multiplyScalar(2));
            }
        });

        if (killCount >= targetKills && !isGameOver) {
            nextLevel();
        }
    }

    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
});

resetGame();
animate();
</script>
</body>
</html>
"""
    html(game_html_tank, height=560)

# ---------------------- 底部卡片署名 ----------------------
st.markdown('<div class="app-footer">', unsafe_allow_html=True)
st.markdown(
    "<div style='color:#1d1d1f; font-size: 15px; line-height:1.7;'>"
    "🏫豫北医学院·王一浩 牛璐鸣 李雨轩<br>"
    "指导教师👩‍🏫：丰依凡"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)
