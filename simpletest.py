import streamlit as st

st.set_page_config( page_title="Simple Connection Test", layout="wide" )

conn = st.connection("snowflake")

