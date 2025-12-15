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

with st.sidebar:
  with st.expander("**Data**"):
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
    st.dataframe(df.select_dtypes("object").describe())

df["BMI Category"] = df["BMI Category"].replace({
    "Normal": "Normalweight",
    "Normal Weight": "Normalweight"
})

if "show_plot" not in st.session_state:
    st.session_state.show_plot_1 = False

if "selected_column" not in st.session_state:
    st.session_state.selected_column = None
with st.sidebar:
  with st.expander("**Visualize how the features are distributed in the dataset**"):
    st.markdown("## Make Your Own Plot")
    select_column= st.selectbox("Select the category feature you want to visualize",df.select_dtypes("object").columns.tolist())
    col1, col2 =st.columns(2)
    with col1:
            plot_btn = st.button("Plot Feature")

    with col2:
            reset_btn = st.button("Reset")
if plot_btn:
    st.session_state.show_plot_1 = True
    st.session_state.selected_column = select_column

if reset_btn:
    st.session_state.show_plot_1 = False
    st.session_state.selected_column = None
  
if st.session_state.show_plot_1 and st.session_state.selected_column:
    fig, ax = plt.subplots()
    df[st.session_state.selected_column].value_counts(normalize=True).plot(kind="bar", xlabel=st.session_state.selected_column, ylabel="Proportion",
                                                      title=f"{st.session_state.selected_column} Proportion(counts in percentage)", color="green", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.success("Plot successfully created")

with st.sidebar:
  with st.expander("**Groupby Table**"):
    st.markdown("## Create Your own Group Table")
    idx_feat =st.multiselect("Select the features for the index", list(df.columns))
    column_feat =st.multiselect("Select the features for your group table column", list(df.columns))
    agg = st.multiselect("Select aggregate(s) function", ["mean", "median", "min", "max", "count", "sum"])
    grp_table = st.button("Show group table")
    
if grp_table and idx_feat and column_feat and agg:
  st.dataframe(
  df.groupby(idx_feat)[column_feat].agg(agg)
        )
if "show_plot" not in st.session_state:
    st.session_state.show_plot = False

if "feat_val" not in st.session_state:
    st.session_state.feat_val = None

if "feature" not in st.session_state:
    st.session_state.feature = None
with st.sidebar:
  with st.expander("**Drill down visualization of features on sleep disorders**"):
    st.session_state.feature = st.selectbox("Choose prefered column", df.select_dtypes("object").nunique().index.tolist(),key="feature_select")
    if st.button("Show unique values"):
            st.session_state.show_plot = False
            st.session_state.feat_val = None
    if st.session_state.feature in df.columns:
      value_list = df[st.session_state.feature].dropna().unique().tolist()
      st.session_state.feat_val = st.selectbox("Select feature value", value_list,key="feat_val_select")
    if st.button("Plot visualization"):
      st.session_state.show_plot = True
if (st.session_state.show_plot and st.session_state.feature in df.columns and st.session_state.feat_val is not None and "Sleep Disorder" in df.columns):
    data_to_plot = (df[df[st.session_state.feature] == st.session_state.feat_val]["Sleep Disorder"].value_counts(normalize=True))
    fig, ax = plt.subplots()
    data_to_plot.plot(
        kind="bar",
        ax=ax,
        xlabel="Sleep Disorders",
        ylabel="Proportion",
        title=f"{st.session_state.feat_val} → Sleep Disorder Distribution")
    st.pyplot(fig)
    st.success("Plot successfully created")

with st.sidebar:
  with st.expander("**Correlation Matrix: Numeric Columns Relationship**"):
    corr_btn = st.button("Show Correlations")
  if corr_btn:
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="Blues")
st.pyplot(plt.gcf())

with st.sidebar:
  with st.expander("**Plot Relationship**"):
    x_var = st.selectbox("Choose the X-axis variable", df.select_dtypes("number").columns.tolist())
    y_var = st.selectbox("Choose the Y-axis variable", df.select_dtypes("number").columns.tolist())
    rel_plot_btn = st.button("Plot Relationship")
if x_var and y_var and rel_plot_btn:
  sns.relplot(df, x=df[x_var], y=d[y_var])
st.pyplot(plt.gcf())

if "df_stack" not in st.session_state:
    st.session_state.df_stack = [df.copy()]  # stack of dataframes
if "df_current" not in st.session_state:
    st.session_state.df_current = df.copy()

ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv  # we will handle division by zero
}
with st.sidebar:
    with st.expander("**Feature Engineering**"):
        col_1 = st.multiselect("Choose feature(s)", list(st.session_state.df_current.columns))
        col_2 = st.multiselect("Choose another feature(s)", list(st.session_state.df_current.columns))
        op = st.selectbox("Choose arithmetic operator", ['*', '/', '+', '-'])
        
        add_button = st.button("Add Feature")
        undo_button = st.button("Undo Last Change")
        reset_button = st.button("Reset to Original Data")

        if add_button and col_1 and col_2 and op:
            st.session_state.df_stack.append(st.session_state.df_current.copy())
            for c1 in col_1:
                for c2 in col_2:
                    new_col = f"{c1}{op}{c2}"
                    try:
                        if op == '/':
                            st.session_state.df_current[new_col] = st.session_state.df_current[c1] / st.session_state.df_current[c2].replace(0, 1e-10)
                        else:
                            st.session_state.df_current[new_col] = ops[op](st.session_state.df_current[c1], st.session_state.df_current[c2])
                    except Exception as e:
                        st.error(f"Error creating column {new_col}: {e}")
            st.success("Dataframe succefully updated")

        if undo_button:
            if len(st.session_state.df_stack) > 1:
                st.session_state.df_current = st.session_state.df_stack.pop()
                st.success("Last change undone.")
            else:
                st.warning("Already at original data!")

        if reset_button:
            st.session_state.df_current = st.session_state.df_stack[0].copy()
            st.session_state.df_stack = [st.session_state.df_stack[0].copy()]
            st.success("Data reset to original.")
        update_btn= st.button("Show updated DataFrame")
if update_btn:
  st.dataframe(st.session_state.df_current)
