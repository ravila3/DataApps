# Import python packages
import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session

# Get the current credentials
connection_parameters = st.secrets["snowflake"]
session = Session.builder.configs(connection_parameters).create()

# Write directly to the app - purely cosmetic headers
st.title("Very Basic GPT Agent")
st.caption("The quickest of examples. :snowflake:")

# Our pre-prompt, we're setting a default but allowing a user to adjust it at a later date
with st.expander("Adjust system prompt"):
    system = st.text_input("System instructions", value="You are a HR expert with deep expertise in gender diversification. Your primary function is to analyse text and assess whether or not individual words are biased towards male or female applicants. Based on the total number of masculine or feminine biased words determine whether the listing leans one way or the other. Your response should give a one sentence summary of your analysis, followed by listing the individual male biased words, followed by the  feminine biased words you identified.")

st.markdown('------') 

# Capture the user input 
prompt = st.text_area('Enter Prompt', height=100).replace("'","")

# Pass the prompt as well as the system pre-prompt to the function, displaying the result
if(st.button('Ask GPT')):
        sql=f"""SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2','{system} this is the text:{prompt}')"""
        st.write(sql)
        result = session.sql(sql).collect()
        st.header('Answer')
        st.write(result[0][0].replace('"',''))