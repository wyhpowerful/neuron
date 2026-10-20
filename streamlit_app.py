import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="帕金森DBS神经动力学仿真可视化平台", layout="wide")
st.title("帕金森DBS神经动力学仿真可视化平台")
st.markdown("📌 上传仿真输出的CSV文件，查看核团电位时序曲线（STN/GPe/GPi等）")

upload_file = st.file_uploader("上传csv仿真数据文件", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.subheader("📋 数据预览")
    st.dataframe(df.head(15), use_container_width=True)

    # 假设第一列是时间
    time_col = df.columns[0]
    signal_cols = df.columns[1:]

    st.sidebar.header("⚙️ 绘图设置")
    selected_cols = st.sidebar.multiselect(
        "选择需要绘制的核团信号",
        options=signal_cols,
        default=list(signal_cols)
    )

    if selected_cols:
        fig = px.line(
            df,
            x=time_col,
            y=selected_cols,
            title="核团膜电位时序变化曲线",
            labels={"value": "膜电位 (mV)", time_col: "时间 (ms)", "variable": "核团"}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("请至少勾选一个核团信号进行绘图")
else:
    st.info("⬆️ 请上传CSV格式仿真结果文件")
