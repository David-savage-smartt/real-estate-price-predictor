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

from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer

import streamlit as st

def wrangle(file):
    df = pd.read_csv(file)  #read the csv file
    
    mask_place = df["place_with_parent_names"].str.contains("Capital Federal")  # *turn place into a variable
    mask_property_type = df["property_type"] == "house"  # * turn prop type to a variable
    mask_price = df["price_approx_usd"] < 400000  # * turn to price to a variable
    low, high = df["surface_area_covered_in_m2"].quantile([0.1,0.9])
    mask_area = df["surface_area_covered_in_m2"].between(low, high)
    df = df[mask_place & mask_price & mask_property_type & mask_area]
    
    df["lat","lon"] = df["lat-lon"].str.split(",", expand=True).astype(float)  #split lat and lon columns
    
    #dropping columns   
    df.drop(columns=[
        #leaky columns
        "price",
        "price_approx_local_currency",
        "price_per_m2",
        "price_usd_per_m2",
        #unnecessary columns- has too much empty cells
        "lat-lon",
        "place_with_parent_names",
        "floor",
        "expenses",
        #columns with high and low cardinality
        "operation",
        "property_type",
        "currency",
        "properati_url",
        #columns with multicolinearity
        "rooms",
        "surface_total_in_m2"
        ], inplace=True)
    
    return df


def load_and_combine_data(file):
    if not file:
        return "invalid  format or nor data was uploaded"
    if file == 1:
        df = wrangle(file[0])
    elif len(file) > 1 and file is not None:
        #files = glob("data/buenos-aires-real-estate-*.csv")  # * find a way to make this less hardcoded
        frames = [wrangle(fil) for fil in file]
        df = pd.concat(frames, ignore_index=True)
    
        

    return df

def model_params_splitter(df):
    target = "price_approx_usd"  # * turn this to a variable
    y = int(len(df) * 0.7)
    print(type(y))
    y_train = df[target].iloc[:y]
    features = ["surface_covered_in_m2", "lat", "lon", "neigborhood"]
    X_train = df[features].iloc[:y]
    
    y_test = df[target].iloc[y:]
    X_test = df[features].iloc[y:]
    
    return {
        "y_train": y_train,
        "X_train": X_train,
        "y_test": y_test,
        "X_test": X_test
        }


def baseline(train):
    train_mean = train.mean()
    pred_baseline = [train_mean] * len(train)
    baseline_mae = mean_absolute_error(train, pred_baseline)
    return {
        "mean": train_mean,
        "pred_baseline": pred_baseline, 
        "baseline_mae": baseline_mae
        }


def modeller(x_train, y_train):
    model = make_pipeline(
        OneHotEncoder(),
        SimpleImputer(),
        Ridge()
        )
    
    return model.fit(x_train, y_train)


def tester(model, test_training_data, y_test_data):
    test_predictions = model.predict(test_training_data)
    test_efficiency = mean_absolute_error(y_test_data, test_predictions)
    
    
    return {
        "test_predictions": test_predictions,
        "test_efficiency": test_efficiency
        }


def main():
    st.title("Price predictor for real-estate data")
    
    uploaded_file = st.file_uploader("add a file or folder here here", accept_multiple_files=True)
    df = load_and_combine_data(uploaded_file)
    
    if isinstance(df, str):
        st.error(df)  # show the error message
        return
    elif df is None:
        st.warning("No data uploaded yet.")
        return
    
    splits = model_params_splitter(df)
    
    st.subheader("Baseline performance with training data")
    training_baseline_data = baseline(splits["y_train"])
    st.metric("training data's mean absolute error", training_baseline_data["baseline_mae"])
    #fig = px.scatter(df, )
    
    st.subheader("Model performance")
    modelled = modeller(splits["X_train"], splits["y_train"])
    model_efficiency = tester(modelled, splits["X_test"], splits["y_test"])
    test_baseline = baseline(splits["y_test"])
    st.metric("baseline of model test data", test_baseline["baseline_mae"])
    st.metric("models MAE", model_efficiency["test_efficiency"])
    

main()