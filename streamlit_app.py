import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="帕金森DBS神经动力学仿真", layout="wide")
st.title("帕金森DBS神经动力学仿真可视化平台")
st.markdown("📌 上传仿真输出的CSV文件，查看核团电位时序曲线（STN/GPe/GPi等）")

upload_file = st.file_uploader("上传csv仿真数据文件", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.subheader("📋 数据预览")
    st.dataframe(df.head(10))

    fig = px.line(df, x=df.columns[0], y=df.columns[1:], title="神经元膜电位变化曲线",
                  labels={"value":"膜电位(mV)", "variable":"信号通道"})
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⬆️ 请上传CSV格式仿真结果文件")
    替换为DBS仿真可视化页面
