import streamlit as st
# import snowflake
import snowflake.connector
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

st.set_page_config(page_title="Alternate Connection Test", layout="wide")

# Initialize the Snowflake session using Streamlit secrets
connection_parameters = st.secrets["snowflake"]
session = Session.builder.configs(connection_parameters).create()

# Query the data using Snowpark
sql = 'SELECT * FROM NOTEBOOK.PUBLIC.WBD_STOCK_PRICE LIMIT 10' # Execute the query and collect the results in a DataFrame 
df_snowpark = session.sql(sql).to_pandas()
# df_snowpark = session.table('NOTEBOOK.PUBLIC.WBD_STOCK_PRICE').limit(10).to_pandas()

# Display the DataFrame in Streamlit
st.write(df_snowpark)

# Close the session
session.close()
