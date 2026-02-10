import streamlit as st
import psycopg2

st.title("PostgreSQL Connection Test")

pg = st.secrets["postgres"]

try:
    conn = psycopg2.connect(
        host=pg["host"],
        port=pg["port"],
        dbname=pg["database"],
        user=pg["user"],
        password=pg["password"]
    )

    st.success("Connected successfully to PostgreSQL!")

    cur = conn.cursor()
    cur.execute("SELECT current_database(), current_user;")
    result = cur.fetchone()
    st.write("Database:", result[0])
    st.write("User:", result[1])

    cur.close()
    conn.close()

except Exception as e:
    st.error(f"Connection failed: {e}")
