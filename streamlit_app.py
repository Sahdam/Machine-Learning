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
from sklearn.utils.class_weight import compute_sample_weight
from streamlit_extras.grid import grid
sns.set_style("dark")


st.title('Machine Learning: Sleep Disorders Classification')

st.info('This app is a machine learning app')

grid1 = grid(1,[5, 5],1,1,1,1, vertical_align ="top")

if "show_data" not in st.session_state:
    st.session_state.show_data = False
with st.sidebar.container():
    d_btn = st.button("**Data**", key="d_btn")
df = pd.read_csv("sleep_health_lifestyle.csv")
df.drop(columns="index", inplace=True)
df.set_index("Person ID", inplace=True)
df.fillna("None", inplace=True)
if d_btn:
    grid1.write("## Sleep, health and Lifestyle Data")
    data_btn = grid1.button("**Show Data**", key="data_btn")
    data_reset =grid1.button("**Show no data**", key="data_reset")
if data_btn:
    st.session_state.show_data = True
if data_reset:
    st.session_state.show_data = False
if st.session_state.show_data:
    grid1.dataframe(df)
    buffer = io.StringIO()
    df.info(buf=buffer)
    grid1.code(buffer.getvalue(), language="text")
    grid1.dataframe(df.describe())
    grid1.dataframe(df.select_dtypes("object").describe())
df["BMI Category"] = df["BMI Category"].replace({"Normal": "Normalweight","Normal Weight": "Normalweight"})

grid2 = grid(1,[4, 2, 1], 1, vertical_align="top")
if "show_plot" not in st.session_state:
    st.session_state.show_plot_1 = False

if "selected_column" not in st.session_state:
    st.session_state.selected_column = None
with st.sidebar.container():
   v_btn = st.button("**Visualize Feature Distribution**", key="v_btn")
if v_btn:
    grid2.write("## Make Your Own Plot")
    select_column= grid2.selectbox("Category feature",df.select_dtypes("object").columns.tolist())
    plot_btn = grid2.button("Plot Feature")
    reset_btn = grid2.button("Reset")
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
        grid2.pyplot(fig)
        plt.close(fig)
        st.success("Plot successfully created")

grid3 = grid(1,[4,5,3],1,1, vertical_align="top")
with st.sidebar.container():
    gr_btn = st.button("**Groupby Table**", key="gr_btn")
if gr_btn:
    grid3.markdown("## Create Your own Group Table")
    idx_feat =grid3.multiselect("Index features", list(df.columns))
    column_feat =grid3.multiselect("Column features", list(df.columns))
    agg = grid3.multiselect("Aggregate(s) function", ["mean", "median", "min", "max", "count", "sum"])
    grp_table = grid3.button("Show group table")
        
    if grp_table and idx_feat and column_feat and agg:
      grid3.dataframe(
      df.groupby(idx_feat)[column_feat].agg(agg)
            )
    if "show_plot" not in st.session_state:
        st.session_state.show_plot = False
    
    if "feat_val" not in st.session_state:
        st.session_state.feat_val = None
    
    if "feature" not in st.session_state:
        st.session_state.feature = None


grid4 = grid([5, 3], [5, 3], 1,1, vertical_align="bottom")
with st.sidebar.container():
   fs_btn = st.button("**Visualize Feature on Sleep Disorder**", key="fs_btn")
if fs_btn:
    grid4.write("## Drill down visualization of features on sleep disorders")
    st.session_state.feature = grid4.selectbox("Choose prefered column", df.select_dtypes("object").nunique().index.tolist(),key="feature_select")
    if grid4.button("Show unique values"):
        st.session_state.show_plot = False
        st.session_state.feat_val = None
    if st.session_state.feature in df.columns:
        value_list = df[st.session_state.feature].dropna().unique().tolist()
        st.session_state.feat_val = grid4.selectbox("Select feature value", value_list,key="feat_val_select")
    if grid4.button("Plot visualization"):
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
        grid4.pyplot(fig)
        plt.close(fig)
        st.success("Plot successfully created")

grid5 = grid([3, 5])
with st.sidebar.container():
    cr_btn = st.button("**Correlation Matrix**", key="cr_btn")
if cr_btn:
    corr_btn = grid5.button("Show Correlations")
    if corr_btn:
        sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="Blues")
        grid5.pyplot(plt.gcf())
    
    if "show_plot" not in st.session_state:
        st.session_state.show_plot_2 = False
    
    if "x_var" not in st.session_state:
        st.session_state.x_var = None
    
    if "y_var" not in st.session_state:
        st.session_state.y_var = None


grid6 = grid([4,4], [3,3],1, vertical_align="top") 
with st.sidebar.container():
    pr_btn =st.button("**Plot Relationship**", key="pr_btn")
if pr_btn:
    x_var = grid6.selectbox("Choose the X-axis variable", df.select_dtypes("number").columns.tolist())
    y_var = grid6.selectbox("Choose the Y-axis variable", df.select_dtypes("number").columns.tolist())
    plot_rel_btn = grid6.button("Plot Relationship")
    reset_btn_1 = grid6.button("Reset Plot")
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
        grid6.pyplot(plt.gcf())


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
grid7 = grid([4,4],[3,2,3],[4,4],1, vertical_align="top")
with st.sidebar.container():
    fe_btn = st.button("**Feature Engineering**", key="fe_btn")
if fe_btn:
    col_1 = grid7.multiselect("Choose feature(s)", list(st.session_state.df_current.columns))
    col_2 = grid7.multiselect("Choose another feature(s)", list(st.session_state.df_current.columns))
    op = grid7.selectbox("Choose arithmetic operator", ['*', '/', '+', '-'])
            
    add_button = grid7.button("Add Feature")
    undo_button = grid7.button("Undo Last Change")
    reset_button = grid7.button("Reset to Original Data")
    
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
    update_btn= grid7.button("Show updated DataFrame")
    if update_btn:
          grid7.dataframe(st.session_state.df_current)
    
    if "df_original" not in st.session_state:
        st.session_state.df_original = df.copy()
    
    if "df_current" not in st.session_state:
        st.session_state.df_current = df.copy()
      
    for key in ["X_train", "X_test", "y_train", "y_test"]:
        if key not in st.session_state:
            st.session_state[key] = None

if "X_train" not in st.session_state:
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None

for k in ["num_col", "cat_col"]:
    if k not in st.session_state:
        st.session_state[k] = []
for m in ["model_lr", "model_dt", "model_rf", "model_gb"]:
    if m not in st.session_state:
        st.session_state[m] = None

grid8 = grid([4,4],[3,3,3],1,1,1, vertical_align="top")
with st.sidebar.container():
    sp_btn = st.button("**Splitting Data into Train and Test**", key="sp_btn")
if sp_btn:
    select_columns = grid8.multiselect("Choose features to drop", st.session_state.df_current.columns.tolist(), key="drop_cols_select")
    testsize = grid8.number_input("Enter Test size (e.g 0.2 for 20%)", min_value=0.1, max_value=0.9, step=0.05, key="test_size")
    drop_btn = grid8.button("Drop Columns", key="drop_btn")
    reset_btn = grid8.button("Reset Dataset", key="reset_btn")
    split_btn = grid8.button("Show Split Data", key="split_btn")
    current_btn = grid8.button("Show Current Dataset", key="current_btn")
    if drop_btn:
        st.session_state.df_current.drop(columns=select_columns, inplace=True)
        for key in ["X_train", "X_test", "y_train", "y_test"]:
            st.session_state[key] = None
    
        st.success("Columns dropped. Please re-split the dataset.")
    if reset_btn:
        st.session_state.df_current = st.session_state.df_original.copy()
        for key in ["X_train", "X_test", "y_train", "y_test"]:
            st.session_state[key] = None
    
        st.success("Dataset restored to original state.")
    if current_btn:
        grid8.subheader("Current Dataset")
        grid8.dataframe(st.session_state.df_current)
    if split_btn:
        if "Sleep Disorder" not in st.session_state.df_current.columns:
            st.error("Target column 'Sleep Disorder' is missing.")
    
        X = st.session_state.df_current.drop(columns="Sleep Disorder")
        y = st.session_state.df_current["Sleep Disorder"]
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=testsize, random_state=42, stratify=y
        )
    
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
    
        st.success("Train-test split created successfully.")
if st.session_state.X_train is not None:
    st.session_state.num_col = (st.session_state.X_train.select_dtypes(include="number").columns.tolist())
    st.session_state.cat_col = (st.session_state.X_train.select_dtypes(include="object").columns.tolist())
    
if st.session_state.num_col or st.session_state.cat_col:
    column_trans = ColumnTransformer(
        [
            ("num", StandardScaler(), st.session_state.num_col),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), st.session_state.cat_col),
        ]
    )
    
    st.session_state.model_lr = Pipeline(
        [
            ("preprocess", column_trans),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    
    st.session_state.model_dt = make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        DecisionTreeClassifier(random_state=True, max_depth=4, class_weight="balanced")
    )
    
    st.session_state.model_rf = make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        RandomForestClassifier(random_state=42, class_weight="balanced_subsample", max_depth=6, n_estimators=20)
    )
    
    sample_weights = compute_sample_weight(class_weight="balanced", y=st.session_state.y_train)
    st.session_state.model_gb= Pipeline(
        steps=[("preprocess", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("gradientboostingclassifier", GradientBoostingClassifier(random_state=42, max_depth=2, n_estimators=40))])

 
if st.session_state.model_lr is not None:

    Xtr = st.session_state.X_train
    ytr = st.session_state.y_train

    st.session_state.model_lr.fit(Xtr, ytr)
    st.session_state.model_dt.fit(Xtr, ytr)
    st.session_state.model_rf.fit(Xtr, ytr)

    sample_weights = compute_sample_weight("balanced", ytr)
    st.session_state.model_gb.fit(
        Xtr,
        ytr,
        model__sample_weight=sample_weights
    )
if st.session_state.model_lr is not None and st.session_state.X_train is not None:
    features = st.session_state.model_lr.named_steps["preprocess"].get_feature_names_out()
    importances = st.session_state.model_lr.named_steps["model"].coef_
    classes = st.session_state.model_lr.named_steps["model"].classes_
    odds_ratio = pd.DataFrame(np.exp(importances),index=classes, columns=features)

    def get_sorted_odds(class_name):
        idx = list(classes).index(class_name)
        return pd.Series(np.exp(importances[idx]), index=features).sort_values()

if st.session_state.model_lr is not None and st.session_state.X_train is not None:
    feat_dt = st.session_state.model_dt.named_steps["onehotencoder"].get_feature_names_out()
    importance_dt = st.session_state.model_dt.named_steps["decisiontreeclassifier"].feature_importances_
    feat_imp_dt = pd.Series(importance_dt, index=feat_dt).sort_values()

if st.session_state.model_lr is not None and st.session_state.X_train is not None:
    feat = st.session_state.model_rf.named_steps["onehotencoder"].get_feature_names_out()
    importance_rf = st.session_state.model_rf.named_steps["randomforestclassifier"].feature_importances_
    feat_imp_rf = pd.Series(importance_rf, index=feat).sort_values()

if st.session_state.model_lr is not None and st.session_state.X_train is not None:
    feat_gb = st.session_state.model_gb.named_steps["preprocess"].get_feature_names_out()
    importance_gb = st.session_state.model_gb.named_steps["gradientboostingclassifier"].feature_importances_
    feat_imp_gb=pd.Series(importance_gb, index=feat_gb).sort_values()

grid9 = grid(1,1,1,1, 1,1,1,1,1, vertical_align="top")
with st.sidebar.container():
    lr_btn = st.button("**Logistic Regression**", key="lr_btn")
if lr_btn:
    feat_imp_btn = grid9.button("**Logistic Regression Analysis**", key="feat_imp_btn")
    if feat_imp_btn:
        grid9.subheader("ODD RATIOS FOR SLEEP DISORDERS") 
        for cls in classes:
            series = get_sorted_odds(cls)
            fig, ax = plt.subplots(1, 2, figsize=(22, 8))
            series.head(10).plot(kind="barh", ax=ax[0])
            ax[0].axvline(1, linestyle="--", color="red")
            ax[0].set_title(f"{cls} — Lowest Odds")
            series.tail(10).plot(kind="barh", ax=ax[1])
            ax[1].axvline(1, linestyle="--", color="red")
            ax[1].set_title(f"{cls} — Highest Odds")
            grid9.pyplot(fig)
        grid9.subheader("Logistic Regression Confusion Matrix")
        ConfusionMatrixDisplay.from_estimator(st.session_state.model_lr, st.session_state.X_test, st.session_state.y_test)
        grid9.pyplot()
        grid9.subheader("Logistic Regression Classification Report")
        grid9.code(classification_report(st.session_state.y_test, st.session_state.model_lr.predict(st.session_state.X_test)))

grid10 = grid(1,1,1,1,1,1, vertical_align = "top")
with st.sidebar.container():
    dtc_btn = st.button("**Decision Tree**", key="dtc_btn")
if dtc_btn:
    dt_btn = grid10.button("**Decision Tree analysis**", key="dt_btn")
    if dt_btn:
      fig1, ax1 = plt.subplots(figsize=(12, 8))
      feat_imp_dt.tail().plot(kind="barh", ax=ax1)
      ax1.set_title("Feature Importance")
      grid10.pyplot(fig1)
      grid10.subheader("Decision Tree Confusion Matrix")
      ConfusionMatrixDisplay.from_estimator(st.session_state.model_dt,st.session_state.X_test, st.session_state.y_test)
      grid10.pyplot()
      grid10.subheader("Decision Tree Classification Report")
      grid10.code(classification_report(st.session_state.y_test, st.session_state.model_dt.predict(st.session_state.X_test)))


grid11 = grid(1,1,1,1,1,1, vertical_align = "top")
with st.sidebar.container():
    rfc_btn = st.button("**Random Forest**", key="rfc_btn")
if rfc_btn:
    rf_btn = grid11.button("**Random Forest analysis**", key="rf_btn")
    if rf_btn:
      fig2, ax2 = plt.subplots(figsize=(12, 8))
      feat_imp_rf.tail().plot(kind="barh", ax=ax2)
      ax2.set_title("Feature Importance")
      grid11.pyplot(fig2)
      grid11.subheader("Random Forest Confusion Matrix")
      ConfusionMatrixDisplay.from_estimator(st.session_state.model_rf,st.session_state.X_test, st.session_state.y_test)
      grid11.pyplot()
      grid11.subheader("Random Forest Classification Report")
      grid11.code(classification_report(st.session_state.y_test, st.session_state.model_rf.predict(st.session_state.X_test)))


grid12 = grid(1,1,1,1,1,1, vertical_align = "top")
with st.sidebar.container():
    gbc_btn = st.button("**Gradient Boosting**", key="gbc_btn")
if gbc_btn:
    gb_btn = grid12.button("**Gradient Boosting analysis**", key="gb_btn")
    if gb_btn:
      fig3, ax3 = plt.subplots(figsize=(12, 8))
      feat_imp_gb.tail().plot(kind="barh", ax=ax3)
      ax3.set_title("Feature Importance")
      grid12.pyplot(fig3)
      grid12.subheader("Gradient Boosting Confusion Matrix")
      ConfusionMatrixDisplay.from_estimator(st.session_state.model_gb,st.session_state.X_test, st.session_state.y_test)
      grid12.pyplot()
      grid12.subheader("Gradient Boosting Classification Report")
      grid12.code(classification_report(st.session_state.y_test, st.session_state.model_gb.predict(st.session_state.X_test)))

