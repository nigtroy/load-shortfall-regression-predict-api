"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
# Libraries for data loading, data manipulation and data visulisation
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
import re
# Libraries for data preparation and model building
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn import linear_model
from fast_ml.feature_engineering import FeatureEngineering_DateTime
import sys
sys.path.append("kuma_utils/")
#from kuma_utils.preprocessing.imputer import LGBMImputer

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #df =  df.drop(columns= 'Unnamed: 0')

  # Dropping Redundant and unsusable features
    snow = sorted([col for col in df.columns if 'snow' in col])
    weather = sorted([col for col in df.columns if 'weather' in col])
    redundant_features = ['Barcelona_rain_3h', 'Seville_rain_3h']
    df.drop(columns = snow, inplace=True)
    df.drop(columns = weather, inplace=True)
    df.drop(columns = redundant_features, inplace=True)

  # remove Rows with pressure values less than 945 and greater than 1055 to subset for outliers
    df = df[df['Barcelona_pressure']>= 945] 
    df = df[df['Barcelona_pressure']<= 1051]

  #Dropping all columns for minimum and maximum temperature readings
    temp = sorted([col for col in df.columns if 'temp' in col])
    min_temp = [ col for col in temp if 'min' in col] 
    max_temp = [ col for col in temp if 'max' in col]
    min_max_temp = min_temp + max_temp
    df = df.drop(columns= min_max_temp)
    
  #Deleting columns withh more than 50% 0 values for testing
    zero_columns = ['Madrid_rain_1h', 'Seville_rain_1h','Barcelona_rain_1h','Bilbao_rain_1h','Seville_pressure']
    df.drop(columns = zero_columns, inplace=True)

  ## converting time object to datetime
    df['time'] = pd.to_datetime(df['time'])
    df['Year'] = df['time'].dt.year
  # month
    df['Month'] = df['time'].dt.month
  # day
    df['Day'] = df['time'].dt.day
  # hour
    df['hour'] = df['time'].dt.hour

  #Rearranging Columns to place time,month,day and our chronologically in dataset
    cols = df.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    df = df[cols]
  
  
  # engineer existing features
    df['Valencia_wind_deg'] = df['Valencia_wind_deg'].str.split('_', expand=True)[1].astype('float') #Split column by '_' and create dataframe from split columns and extract second column
    df = df.set_index("time", drop=True, append=False, inplace=False, verify_integrity=False)
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    # ------------------------------------------------------------------------
    predict_vector = df
    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
