import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from streamlit_extras.grid import grid
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
sns.set_style("dark")
st.set_page_config(page_title="Sleep Disorder ML App", layout="wide")

# SESSION STATE INITIALIZATION
if "page" not in st.session_state:
    st.session_state.page = "Data"

if "df_original" not in st.session_state:
    df = pd.read_csv("sleep_health_lifestyle.csv")
    df.drop(columns="index", inplace=True)
    df.set_index("Person ID", inplace=True)
    df.fillna("None", inplace=True)
    df["BMI Category"] = df["BMI Category"].replace(
        {"Normal": "Normalweight", "Normal Weight": "Normalweight"}
    )
    st.session_state.df_original = df.copy()
    st.session_state.df_current = df.copy()

for k in ["X_train", "X_test", "y_train", "y_test"]:
    st.session_state.setdefault(k, None)

for m in ["model_lr", "model_dt", "model_rf", "model_gb"]:
    st.session_state.setdefault(m, None)

# SIDEBAR NAVIGATION
with st.sidebar:
    st.title("Navigation")
    st.session_state.page = st.radio(
        " ",[
            "Data",
            "EDA",
            "Feature Engineering",
            "Train / Test Split",
            "Model Training",
            "Model Evaluation",
            "Prediction"
        ],
    )

# PAGE 1 — DATA
def data_page():
    st.header("Sleep, Health and Lifestyle Dataset")

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(st.session_state.df_current)

    with col2:
        buffer = io.StringIO()
        st.session_state.df_current.info(buf=buffer)
        st.code(buffer.getvalue(), language="text")

        st.subheader("Numerical Summary")
        st.dataframe(st.session_state.df_current.describe())

        st.subheader("Categorical Summary")
        st.dataframe(
            st.session_state.df_current.select_dtypes("object").describe()
        )

# PAGE 2 — EDA
grid_obj = grid(1,1,1,[5, 3], 1, 1, vertical_align="top")
def eda_page(grid_obj):
    grid_obj.header("Exploratory Data Analysis")

    cat_features = st.session_state.df_current.select_dtypes("object").columns.tolist()

    feature = grid_obj.selectbox("Select categorical feature to visualize", cat_features, key="eda_feature_select")

    if feature:
        fig, ax = plt.subplots(figsize=(6, 4))
        st.session_state.df_current[feature].value_counts(normalize=True).plot(
            kind="bar", ax=ax, color="skyblue"
        )
        ax.set_title(f"{feature} Distribution")
        grid_obj.pyplot(fig)
        plt.close(fig)

    if feature:
        unique_values = st.session_state.df_current[feature].dropna().unique().tolist()
        value = grid_obj.selectbox("Select value to examine its Sleep Disorder distribution", unique_values, key="eda_feature_val_select")

        plot_btn = grid_obj.button("Plot Sleep Disorder Distribution", key="eda_feature_plot_btn")

        if plot_btn and value:
            data_to_plot = st.session_state.df_current[st.session_state.df_current[feature] == value]["Sleep Disorder"].value_counts(normalize=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            data_to_plot.plot(
                kind="bar",
                ax=ax,
                xlabel="Sleep Disorders",
                ylabel="Proportion",
                title=f"{value} → Sleep Disorder Distribution",
                color="skyblue"
            )
            grid_obj.pyplot(fig)
            plt.close(fig)
            st.success("Plot successfully created")
    
# PAGE 3 — FEATURE ENGINEERING
def feature_engineering_page():
    st.header("Feature Engineering")
    st.subheader("Current Dataset")
    st.dataframe(st.session_state.df_current, use_container_width=True)
    df = st.session_state.df_current

    st.markdown("### Drop Features")
    drop_cols = st.multiselect(
        "Select columns to drop",
        st.session_state.df_current.columns
    )
    col1, col2 = st.columns(2)
    with col1:
        c1 = st.selectbox("Feature 1", df.columns)
        op = st.selectbox("Operation", ["+", "-", "*", "/"])
        c2 = st.selectbox("Feature 2", df.columns)

    with col2:
        if st.button("Add Feature"):
            new_col = f"{c1}{op}{c2}"
            try:
                if op == "/":
                    df[new_col] = df[c1] / df[c2].replace(0, np.nan)
                elif op == "+":
                    df[new_col] = df[c1] + df[c2]
                elif op == "-":
                    df[new_col] = df[c1] - df[c2]
                elif op == "*":
                    df[new_col] = df[c1] * df[c2]
                st.success(f"Added {new_col}")
            except Exception as e:
                st.error(e)
    if st.button("Apply Drop"):
        st.session_state.df_current = st.session_state.df_current.drop(columns=drop_cols)
        st.success("Features updated")

    st.subheader("Updated Dataset Preview")
    st.dataframe(st.session_state.df_current.head())
    if st.button("Reset Dataset"):
        st.session_state.df_current = st.session_state.df_original.copy()
        st.success("Dataset reset")

# PAGE 4 — SPLIT
def split_page():
    st.header("Train / Test Split")

    test_size = st.slider("Test size", 0.1, 0.5, 0.2)

    if st.button("Split Data"):
        df = st.session_state.df_current
        X = df.drop(columns="Sleep Disorder")
        y = df["Sleep Disorder"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        st.session_state.update({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        })

        st.success("Dataset split completed")
        if "feature_columns" not in st.session_state:
            st.session_state.feature_columns = X_train.columns.tolist()

    if "X_train" in st.session_state:
        st.subheader("Train Data")
        st.dataframe(st.session_state.X_train)
        st.subheader("Train Target")
        st.dataframe(st.session_state.y_train)
        st.subheader("Test Data")
        st.dataframe(st.session_state.X_test)
        st.subheader("Test Target")
        st.dataframe(st.session_state.y_test)

# PAGE 5 — TRAINING
def training_page():
    st.header("Modeling & Feature Importance")
    st.info("**Select the model of your choice and click on train models**")

    if st.session_state.X_train is None:
        st.warning("Please split data first")
        return

    Xtr = st.session_state.X_train
    ytr = st.session_state.y_train

    num_cols = Xtr.select_dtypes("number").columns
    cat_cols = Xtr.select_dtypes("object").columns

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    st.session_state.model_lr = Pipeline(
        [
            ("prep", preprocess),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )

    st.session_state.model_dt = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        DecisionTreeClassifier(max_depth=4, class_weight="balanced"),
    )

    st.session_state.model_rf = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        RandomForestClassifier(n_estimators=50, class_weight="balanced"),
    )

    sample_w = compute_sample_weight("balanced", ytr)
    st.session_state.model_gb = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        GradientBoostingClassifier(n_estimators=50),
    )

    if st.button("Train Models"):
        st.session_state.model_lr.fit(Xtr, ytr)
        st.session_state.model_dt.fit(Xtr, ytr)
        st.session_state.model_rf.fit(Xtr, ytr)
        st.session_state.model_gb.fit(Xtr, ytr, gradientboostingclassifier__sample_weight=sample_w)
        st.success("Models trained successfully")
        
    if st.checkbox("Logistic Regression Feature Importance"):
        try:
            features = st.session_state.model_lr.named_steps["prep"].get_feature_names_out()
            coef = st.session_state.model_lr.named_steps["model"].coef_
            odds = pd.Series(np.exp(coef[0]), index=features).sort_values()
            fig, ax = plt.subplots(figsize=(8, 10))
            odds.tail(10).plot(kind="barh", ax=ax)
            ax.axvline(1, color="red", linestyle="--")
            ax.set_title("Top Logistic Regression Odds Ratios")
            st.pyplot(fig)
            fig2, ax = plt.subplots(figsize=(8, 10))
            odds.head(10).plot(kind="barh", ax=ax)
            ax.axvline(1, color="red", linestyle="--")
            ax.set_title("Top Logistic Regression Odds Ratios")
            st.pyplot(fig2)
        except NotFittedError:
            st.warning("Model is not trained yet. Train the model first to see feature importance.")
            return
    def plot_tree_importance(model, title):
        feat = model.named_steps[list(model.named_steps.keys())[0]].get_feature_names_out()
        imp = model.named_steps[list(model.named_steps.keys())[1]].feature_importances_
        series = pd.Series(imp, index=feat).sort_values().tail(10)
        fig, ax = plt.subplots(figsize=(8, 10))
        series.plot(kind="barh", ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
    if st.checkbox("Decision Tree Importance"):
        try:
            plot_tree_importance(st.session_state.model_dt, "Decision Tree Feature Importance")
        except NotFittedError:
            st.warning("Model is not trained yet. Train the model first to see feature importance.")
            return

    if st.checkbox("Random Forest Importance"):
        try:
            plot_tree_importance(st.session_state.model_rf, "Random Forest Feature Importance")
        except NotFittedError:
            st.warning("Model is not trained yet. Train the model first to see feature importance.")
            return
    
    if st.checkbox("Gradient Boosting Importance"):
        try:
            plot_tree_importance(st.session_state.model_gb, "Gradient Boosting Feature Importance")
        except NotFittedError:
            st.warning("Model is not trained yet. Train the model first to see feature importance.")
            return

# PAGE 6 — EVALUATION
def evaluation_page():
    st.header("Model Evaluation")

    if st.session_state.model_lr is None:
        st.warning("Train models first")
        return

    model_name = st.selectbox(
        "Choose model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
    )

    model_map = {
        "Logistic Regression": st.session_state.model_lr,
        "Decision Tree": st.session_state.model_dt,
        "Random Forest": st.session_state.model_rf,
        "Gradient Boosting": st.session_state.model_gb,
    }

    try:
        model = model_map[model_name]

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(
            model,
            st.session_state.X_test,
            st.session_state.y_test,
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)
    
        st.code(
            classification_report(
                st.session_state.y_test,
                model.predict(st.session_state.X_test),
            )
        )
    except NotFittedError:
            st.warning("Model is not trained yet. Train the model first to see feature importance.")
            return

# Make Prediction
def prediction_page():
    st.header("Predict Sleep Disorder")
    def align_features(df_new):
        return df_new.reindex(
            columns=st.session_state.feature_columns,
            fill_value=0
        )
    mode = st.radio(
    "Prediction Method",
    ["Manual Entry", "Upload CSV"])
    if mode == "Manual Entry":
        st.subheader("Manual Input")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 10, 100)
        occupation = st.text_input("Occupation")
        sleep_duration = st.number_input("Sleep Duration", 1, 10)
        sleep_quality = st.number_input("Quality of Sleep", 1, 10)
        physical_act = st.number_input("Physical Activity Level", 1, 10)
        stress_level = st.number_input("Stress Level", 1, 10)
        bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
        blood_pre = st.text_input("Blood Pressure")
        hr = st.number_input("Heart Rate", 30, 200)
        daily_step = st.number_input("Daily Steps", 0, 30000, step=500)
    
        data_new = pd.DataFrame({
            "Gender": [gender],
            "Age": [age],
            "Occupation": [occupation],
            "Sleep Duration": [sleep_duration],
            "Quality of Sleep": [sleep_quality],
            "Physical Activity Level": [physical_act],
            "Stress Level": [stress_level],
            "BMI Category": [bmi],
            "Blood Pressure": [blood_pre],
            "Heart Rate": [hr],
            "Daily Steps": [daily_step],
        })
        data_new = align_features(data_new)
    if mode == "Upload CSV":

        st.subheader("Upload CSV File")
    
        uploaded_file = st.file_uploader(
            "Upload CSV with same structure as training data",
            type=["csv"]
        )
    
        if uploaded_file:
            df_uploaded = pd.read_csv(uploaded_file)
    
            st.write("📄 Uploaded Data Preview")
            st.dataframe(df_uploaded.head())
    
            df_uploaded = align_features(df_uploaded)
    
    model_name = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    )
    
    model_map = {
        "Logistic Regression": st.session_state.model_lr,
        "Decision Tree": st.session_state.model_dt,
        "Random Forest": st.session_state.model_rf,
        "Gradient Boosting": st.session_state.model_gb,
    }
    
    model = model_map[model_name]
    if st.button("Predict"):
        if model is None:
                st.warning("Train the model first.")
                return
        
        if mode == "Manual Entry":
        try:
            pred = model.predict(data_new)[0]
            proba = model.predict_proba(data_new).max()
            
            st.success(f"Prediction: **{pred}**")
            st.info(f"Confidence: **{proba:.2%}**")
        except NotFittedError:
            st.warning("Model is not trained yet. Train the model first to see feature importance.")
            return
        
        else:
            try:
                predictions = model.predict(df_uploaded)
                df_uploaded["Prediction"] = predictions
            
                st.success("Batch prediction completed")
                st.dataframe(df_uploaded)
            
                st.download_button(
                    "Download Results",
                    df_uploaded.to_csv(index=False),
                    "sleep_predictions.csv"
                    )
            except NotFittedError:
                st.warning("Model is not trained yet. Train the model first to see feature importance.")
                return

    
# ROUTER

if st.session_state.page == "Data":
    data_page()
elif st.session_state.page == "EDA":
    eda_page(grid_obj)
elif st.session_state.page == "Feature Engineering":
    feature_engineering_page()
elif st.session_state.page == "Train / Test Split":
    split_page()
elif st.session_state.page == "Model Training":
    training_page()
elif st.session_state.page == "Model Evaluation":
    evaluation_page()
elif st.session_state.page == "Prediction":
    prediction_page()
