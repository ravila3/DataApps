import streamlit as st

st.set_page_config( page_title="Simple Connection Test", layout="wide" )

conn = st.connection("snowflake")
# df=conn.query('select * from NOTEBOOK.PUBLIC.WBD_STOCK_PRICE limit 10')
# st.write(df)

st.write('test passed')
