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
import numpy as np 
import glob as glob
#import matplotlib.pyplot as plt
import plotly.express as px

from pandas.api.types import is_numeric_dtype, is_categorical_dtype

from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from scipy.stats import pearsonr
import pingouin as pg

import streamlit as st


#functions to enhance the modularity

def wrangle(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
        df.dropna()
        
    else :
        raise ValueError("Unsupported file type, please submit a .csv or .xlsx (excel) file")
    
    return df


def  listify_columns(df):
    column_list = list(df.columns) 
    
    return column_list


def det_task_type(df,target_column):
    column_type = df[target_column].dtype
    unique_entries = df[target_column].nunique()
    
    if column_type == "object" or unique_entries < 10:
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


def apply_stratify(task_type, target):
    if task_type == "Classification":
        stratify = target
    elif task_type == "Regression":
        stratify = None 
    
    return stratify


def decode_data(coded_data, label_map):
    decoded_data = [label_map[code] for code in coded_data]
    
    return decoded_data


def obtain_cors(df, feature_columns, target_column):
    target_is_num = np.issubdtype(df[target_column].dtype, np.number)
    
    for feature_column in feature_columns:
        feature_is_num = np.issubtype(df[feature_column].dtype, np.number)
        
        if target_is_num and feature_is_num:
            cors_method = scipy.stats.pearsonr
        elif target_is_num and not feature_is_num:
            cors_method = sklearn.feature_selection.f_classif
        elif not target_is_num and feature_is_num:
            cors_method = sklearn.feature_selection.f_classif
        else:
            cors_method =  pg.cramers_v
        
        return cors_method


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
            return 
        
        task_type = det_task_type(df, target_column)
        stratify_val = apply_stratify(task_type, df[target_column])
        X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df[target_column], test_size = 0.3, random_state = 42, stratify=stratify_val)
        
        if task_type =="Classification":
            y_train = y_train.astype("category")
            y_test = y_test.astype("category")
            
            label_map = dict(enumerate(y_train.cat.categories))
            
            y_train = y_train.cat.codes
            y_test = y_test.cat.codes
            
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
            decoded_ytest = decode_data(y_test, label_map)
            decoded_ypred = decode_data(y_pred, label_map)
            viz_df = pd.DataFrame({"Actual": decoded_ytest, "Predicted": decoded_ypred})
            fig = px.histogram(viz_df, x="Actual", color="Predicted", barmode="group",
                               title="Actual vs Predicted Class Distribution")
            st.plotly_chart(fig)

        if (task_type =="Regression" and acc < 1) or (task_type=="Classification" and acc<0.75):
            st.write("From the data, we were able to understand the data you are working with as shown by the accuracy of our model, would you like to see some unique insights from the data, using understandable visualizations")
            insight_button = st.button("Click here for insights")
            
            if insight_button:
                for feat in feature_columns:
                    fig = px.scatter(df, x =feat,
                                     y=target_column, title="Interactive Charts for insights"
                        )
                
                st.write("the 3 features with the most correlation with the target are: ")
                
                cors = obtain_cors(df, feature_columns, target_column)
                
                
                
                #identify the feature columns, 
                #plot them against the target columns
                #show which columns are the most relevant for the target column // like the 80/20 rule
                
                pass
            


main()