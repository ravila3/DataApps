import streamlit as st
import psycopg2

st.title("PostgreSQL Connection Test")

try:
    conn = psycopg2.connect(
        host="localhost",
        database="financial_data",
        user="financial_app",
        password="Rafman11"
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
