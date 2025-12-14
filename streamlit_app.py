import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import operator

st.title('Machine Learning: Sleep Disorders Classification')

st.info('This app is a machine learning app')

with st.expander("Data"):
  st.write("## Sleep, health and Lifestyle Data")
  df = pd.read_csv("sleep_health_lifestyle.csv")
  df.drop(columns="index", inplace=True)
  df.set_index("Person ID", inplace=True)
  df.fillna("None", inplace=True)
  df
  buffer = io.StringIO()
  df.info(buf=buffer)
  st.code(buffer.getvalue(), language="text")
  st.dataframe(df.describe())

df["BMI Category"] = df["BMI Category"].replace({
    "Normal": "Normalweight",
    "Normal Weight": "Normalweight"
})

with st.expander("Visualize how the features are distributed in the dataset"):
  st.markdown("## Make Your Own Plot")
  select_column= st.selectbox("Select the category feature you want to visualize",list(df.select_dtypes("object").nunique().index))
  fig, ax = plt.subplots()
  df[select_column].value_counts(normalize=True).plot(kind="bar", xlabel=f"{select_column}", ylabel="Proportion",
                                                    title=f"{select_column} Proportion(counts in percentage)", color="green", ax=ax)
  st.pyplot(fig)
  st.success("Plot successfully created")

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

ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv  # we will handle division by zero
}

with st.sidebar:
  with st.expander("Select Feature for feature engineering"):
    col_1 = st.multiselect("Choose feature", list(df.columns))
    col_2= st.multiselect("Choose another feature", list(df.columns))
    op = st.selectbox("Choose arithmetic operator", ['*', '/', '+', '-'])
    add_button = st.button("Add Feature")
    undo_button = st.button("Undo Last Change")
    if add_button and col_1 and col_2 and op:
            st.session_state.df_previous = st.session_state.df_current.copy()
            for c1 in col_1:
                for c2 in col_2:
                    new_col_name = f"{c1}{op}{c2}"
                    try:
                        if op == '/':
                            st.session_state.df_current[new_col_name] = st.session_state.df_current[c1] / st.session_state.df_current[c2].replace(0, 1e-10)
                        else:
                            st.session_state.df_current[new_col_name] = ops[op](st.session_state.df_current[c1], st.session_state.df_current[c2])
                    except Exception as e:
                        st.error(f"Error creating column {new_col_name}: {e}")
        
    if undo_button:
      if "df_previous" in st.session_state:
                st.session_state.df_current = st.session_state.df_previous.copy()
                st.success("Last change undone.")
      else:
                st.warning("Nothing to undo.")

st.write("**Updated DataFrame:**", st.session_state.df_current)
