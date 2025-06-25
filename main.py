'''
1. prepare data
   explore
   split
2. build model
   iterate
   evaluate
3. communicate results
'''

#prepare data - import 
import pandas as pd
import glob as glob
#import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import streamlit as st

def wrangle(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else :
        raise ValueError("Unsupported file type, please submit a .csv or .xlsx (excel) file")
    
    return df


def  listify_columns(df):
    column_list = list(df.columns) 
    
    return column_list


def det_task_type(df,target_column):
    column_type = df[target_column].dtype
    unique_entries = df[target_column].nunique()
    
    if column_type == object() or bool() or unique_entries < 10:
        task_type = "Classification"
    else:
        task_type = "Regression"
   
    return task_type


def get_model(task_type, numerical_cols, categorical_cols):
    if task_type == "Regression":
        model = Ridge()
    else:
        model = LogisticRegression(max_iter=1000)
    
    transformers = []
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    if numerical_cols:
        transformers.append(("num", SimpleImputer(), numerical_cols))

    
    preprocessor = ColumnTransformer(
        transformers= transformers
        )
    
    pipeline = make_pipeline(
        preprocessor, model
        )
    
    return pipeline

def main():
    st.title("Price predictor for real-estate data")
    
    uploaded_file = st.file_uploader("add a file or folder here here", accept_multiple_files=True)
    dfs = [wrangle(file) for file in uploaded_file]
    df = pd.concat(dfs, ignore_index=True)

    column_list = listify_columns(df)
    st.write(column_list)
    
    with st.form("model form - #coming up with names is a thing?"):
        st.write("choose columns for modelling")
        target_column = st.selectbox("select target column", column_list)
        feature_columns = st.multiselect("select feature columns", column_list) 
        
        submitted = st.form_submit_button("submit")
        
    if submitted:
        if not target_column or not feature_columns:
            st.warning("pick an target and your features")
            return #why not use break and continue
        
        X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df[target_column], test_size = 0.3, random_state = 42)
        task_type = det_task_type(df, target_column)
        st.markdown(f"this is {task_type} task")
    
        categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
        numerical_cols = X_train.select_dtypes(include="number").columns.tolist()
        model = get_model(task_type, numerical_cols, categorical_cols)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task_type == "Regression":
            acc = mean_absolute_error(y_test, y_pred)
            st.metric("mean Absolute Error: ", round(acc, 2))
            
            viz_df = pd.DataFrame({"Actual":y_test, "Predicted": y_pred})
            fig = px.scatter(viz_df, x="Actual", y="Predicted",
                             title="Actual vs Predicted",
                             labels={"x": "Actual values", "y":"Predicted values"})
            st.plotly_chart(fig)
            
        else:
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{round(acc * 100,2)}%")
            
            viz_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            fig = px.histogram(viz_df, x="Actual", color="Predicted", barmode="group",
                               title="Actual vs Predicted Class Distribution")
            st.plotly_chart(fig)


main()