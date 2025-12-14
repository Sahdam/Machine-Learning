import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.title('Machine Learning: Sleep Disorders Classification')

st.info('This app is a machine learning app')

with st.expander("Data"):
  st.write("## Sleep, health and Lifestyle Data")
  df = pd.read_csv("sleep_health_lifestyle.csv")
  df.drop(columns="index", inplace=True)
  df.set_index("Person ID", inplace=True)
  df.fillna("None", inplace=True)
  df

df["BMI Category"] = df["BMI Category"].replace({
    "Normal": "Normalweight",
    "Normal Weight": "Normalweight"
})

with st.expander("Visualize how the features are distributed in the dataset"):
  st.markdown("## Make Your Own Plot")
  select_column= st.selectbox("Select the category feature you want to visualize",
                              ["Gender", "Occupation", "BMI Category", "Blood Pressure"])
  fig, ax = plt.subplots()
  df[select_column].value_counts(normalize=True).plot(kind="bar", xlabel=f"{select_column}", ylabel="Proportion",
                                                    title=f"{select_column} Proportion(counts in percentage)", color="red")
  st.pyplot(fig)
