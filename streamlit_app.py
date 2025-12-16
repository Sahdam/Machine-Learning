import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import operator
sns.set_style("dark")

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

if "show_plot" not in st.session_state:
    st.session_state.show_plot_2 = False

if "x_var" not in st.session_state:
    st.session_state.x_var = None

if "y_var" not in st.session_state:
    st.session_state.y_var = None
  
with st.sidebar:
  with st.expander("**Plot Relationship**"):
    x_var = st.selectbox("Choose the X-axis variable", df.select_dtypes("number").columns.tolist())
    y_var = st.selectbox("Choose the Y-axis variable", df.select_dtypes("number").columns.tolist())
    col_1, col_2 =st.columns(2)
    with col_1:
            plot_rel_btn = st.button("Plot Relationship")

    with col_2:
            reset_btn_1 = st.button("Reset Plot")
if plot_rel_btn:
    st.session_state.show_plot_2 = True
    st.session_state.x_var = x_var
    st.session_state.y_var = y_var

if reset_btn_1:
    st.session_state.show_plot_2 = False
    st.session_state.x_var = None
    st.session_state.y_var = None
  
if st.session_state.x_var and st.session_state.y_var and st.session_state.show_plot_2:
  sns.regplot(data=df, x=df[x_var], y=df[y_var], ci=None,
              color="red")
  plt.title(f"Relation Between{st.session_state.x_var} and {st.session_state.y_var}")
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

if "df_original" not in st.session_state:
    st.session_state.df_original = df.copy()

if "df_current" not in st.session_state:
    st.session_state.df_current = df.copy()

with st.sidebar:
    with st.expander("**Splitting Data into Train and Test**"):
        select_columns = st.multiselect("Choose features to drop", st.session_state.df_current.columns.tolist(), key="drop_cols_select")
        testsize = st.number_input("Enter Test size (e.g 0.2 for 20%)", min_value=0.1, max_value=0.9, step=0.05, key="test_size")
        drop_btn = st.button("Drop Columns", key="drop_btn")
        reset_btn = st.button("Reset Dataset", key="reset_btn")
        show_split_btn = st.button("Show Split Data", key="show_split_btn")
        current_btn = st.button("Current Dataset", key="current_btn")
if drop_btn and select_columns:
    st.session_state.df_current = st.session_state.df_current.drop(columns=select_columns)
    st.success("Columns dropped successfully")
if reset_btn:
    st.session_state.df_current = st.session_state.df_original.copy()
    st.success("Dataset has been reset to original state")
if current_btn:
  st.subheader("Current Dataset")
  st.dataframe(st.session_state.df_current)
if show_split_btn:
    df_working = st.session_state.df_current
    if "Sleep Disorder" not in df_working.columns:
        st.error("Target column 'Sleep Disorder' is missing")
    else:
        X = df_working.drop(columns="Sleep Disorder")
        y = df_working["Sleep Disorder"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testsize), random_state=42)
        st.subheader("X_train")
        st.dataframe(X_train)
        st.subheader("y_train")
        st.dataframe(y_train)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test


X_train = st.session_state.X_train
y_train = st.session_state.y_train
num_col = X_train.select_dtypes(include="number").columns.tolist()
cat_col = X_train.select_dtypes(include="object").columns.tolist()

column_trans = ColumnTransformer(
    [
        ("num", StandardScaler(), num_col),
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False), cat_col)
    ]
)
model_lr = Pipeline(
    [
        ("preprocess", column_trans),
        ("model", LogisticRegression(class_weight="balanced",  max_iter=1000))
    ]
)
if "X_train" not in st.session_state:
    st.warning("Split the data first.")
    st.stop()
model_lr.fit(X_train, y_train)

features = model_lr.named_steps["preprocess"].get_feature_names_out()
importances = model_lr.named_steps["model"].coef_
classes = model_lr.named_steps["model"].classes_
odds_ratio = pd.DataFrame(np.exp(importances),index=classes, columns=features)

def get_sorted_odds(class_name):
    idx = list(classes).index(class_name)
    return pd.Series(np.exp(importances[idx]), index=features).sort_values()

insomnia_odds_sorted = get_sorted_odds("Insomnia")
none_odds_sorted = get_sorted_odds("None")
sa_odds_sorted = get_sorted_odds("Sleep Apnea")

if "X_test" not in st.session_state:
  st.warning("Please split the data first.")
  st.stop()
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

with st.sidebar:
  with st.expander("**Logistic Regression**"):
    feat_imp_btn = st.button("**Feature Importances (Odds Ratios)**", key="feat_imp_btn")
if feat_imp_btn:
  for title, series in {
        "Insomnia": insomnia_odds_sorted,
        "None": none_odds_sorted,
        "Sleep Apnea": sa_odds_sorted
    }.items():

        fig, ax = plt.subplots(1, 2, figsize=(25, 8))

        series.head(10).plot(kind="barh", ax=ax[0])
        ax[0].axvline(1, linestyle="--", color="red")
        ax[0].set_title(f"{title}: Lowest Odds")

        series.tail(10).plot(kind="barh", ax=ax[1])
        ax[1].axvline(1, linestyle="--", color="red")
        ax[1].set_title(f"{title}: Highest Odds")

        st.pyplot(fig)
  ConfusionMatrixDisplay.from_estimator(model_lr, X_test, y_test)
  st.pyplot()
  st.code(classification_report(y_test, model_lr.predict(X_test)))
