import streamlit as st

st.title('Machine Learning: Sleep Disorders Classification')

st.info('This app is a machine learning app')

df = pd.read_csv("master/sleep_health_lifestyle.csv")
df
