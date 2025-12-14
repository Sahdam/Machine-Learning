import streamlit as st
import pandas as pd
st.title('Machine Learning: Sleep Disorders Classification')

st.info('This app is a machine learning app')

with st.expander("Data"):
  st.write("## Sleep, health and Lifestyle Data")
  df = pd.read_csv("sleep_health_lifestyle.csv")
  df.drop(columns="index", inplace=True)
  df.set_index("Person ID", inplace=True)
  df.fillna("None", inplace=True)
  df
