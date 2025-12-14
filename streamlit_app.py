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
  select_column= st.selectbox("Select the category feature you want to visualize",list(df.select_dtypes("object").unique()))
  fig, ax = plt.subplots()
  df[select_column].value_counts(normalize=True).plot(kind="bar", xlabel=f"{select_column}", ylabel="Proportion",
                                                    title=f"{select_column} Proportion(counts in percentage)", color="green", ax=ax)
  st.pyplot(fig)

with st.expander("Groupby Table"):
  st.markdown("## Create Your own Group Table")
  idx_feat =st.multiselect("Select the features for the index", list(df.columns))
  column_feat =st.multiselect("Select the features for your group table column", list(df.columns))
  agg = st.multiselect("Select aggregate(s) function", ["mean", "median", "min", "max", "count", "sum"])
  
  if idx_feat and column_feat and agg:
      st.dataframe(
          df.groupby(idx_feat)[column_feat].agg(agg)
      )
  else:
      st.warning("Please select at least one index, one column, and one aggregate function.")
