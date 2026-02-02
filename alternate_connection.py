import streamlit as st
import snowflake.connector
import pandas as pd

st.set_page_config( page_title="Alternate Connection Test", layout="wide" )

conn = snowflake.connector.connect(**st.secrets["snowflake"])
# cur = conn.cursor()
# data_dict=cur.execute('select * from NOTEBOOK.PUBLIC.WBD_STOCK_PRICE limit 10').fetchall()
# st.write(data_dict)
# columns = [desc[0] for desc in cur.description]
# st.write(columns)
# df=pd.DataFrame(data_dict,columns=columns)

df=pd.read_sql('select * from NOTEBOOK.PUBLIC.WBD_STOCK_PRICE limit 10',conn)
st.write(df)

st.write('test passed')

# my_cnx = snowflake.connector.connect(**st.secrets["snowflake"])
#     my_cur = my_cnx.cursor()
#     # run a snowflake query and put it all in a var called my_catalog
#     my_cur.execute("select * from SWEATSUITS")
#     my_catalog = my_cur.fetchall()
#     st.dataframe(my_catalog)