import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st

def arno_wrangle(filepath):
    #Load to dataframe
    df = pd.read_csv(filepath)

    #Conveting date column to Datetime datatype
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    #Dropping Null value dominated years and fill left Nan rows
    df.drop(df[df.Date < pd.to_datetime("01/01/2004", dayfirst=True)].index, inplace=True)
    df.drop(df[df.Date > pd.to_datetime("01/07/2007", dayfirst=True)].index, inplace=True)
    df.fillna(0, inplace=True)

    #Setting date column as index
    df = df.set_index("Date")

    return df

def split(data, train_frac=0.7, val_frac=0.30):
    
    # contiguous split
    train_delim = int(train_frac * len(data))
    val_delim = int((train_frac + val_frac) * len(data))

    indices = data.index
    start, end = indices[0], indices[train_delim - 1]
    train = data[start:end]
    
    start, end = indices[train_delim], indices[val_delim - 1]
    val = data[start:end]
    
    if val_delim < len(data):
        start = indices[val_delim]
        test = data[start:]
        
    else:
        test = []
    
    return train, val, test

def get_train_features(dataframe, max_lag):
    
    columnVals = dataframe.columns.map(lambda x: x.startswith("Rainfall"))
    df_rainfall = dataframe.filter(dataframe.columns[columnVals], axis=1)
    X_rain = df_rainfall.to_numpy()
    
    # split rainfall into two components, one component is a function of rainfall in region 1: columns 0 to 5 
    # and the second component is a function of rainfall in region 2: columns 6 to 13 
    rainfall_1 = np.linalg.norm(X_rain[:, 0:6], ord=1, axis=-1) / 100
    rainfall_2 = np.linalg.norm(X_rain[:, 6:], ord=1, axis=-1) / 100

    # shift by max_lag + 1 because we are matching rainfall with hydrometry difference
    data = {"rainfall_1": rainfall_1[max_lag+1:],
            "rainfall_2": rainfall_2[max_lag+1:]}
        
    for i in range(1, max_lag + 1):
        # use max_lag-i + 1 as starting index for series delayed by i steps
        data["rainfall_2_lag_{}".format(i)] = rainfall_2[max_lag-i+1:-i]
        data["rainfall_1_lag_{}".format(i)] = rainfall_1[max_lag-i+1:-i]
        
    hydrometry = dataframe["Hydrometry_Nave_di_Rosano"]
    target = pd.DataFrame({"hydrometry_diff": (hydrometry - hydrometry.shift(1))[max_lag+1:]}, 
                          index=dataframe.index[max_lag+1:])
    
    exog = pd.DataFrame(data=data, index=target.index)
    
    return exog, target


def get_val_features(train_df, val_df, max_lag):
    
    columnVals = train_df.columns.map(lambda x: x.startswith("Rainfall"))
    
    df_rainfall = train_df[-max_lag:].filter(train_df.columns[columnVals], axis=1)
    df_rainfall = pd.concat([df_rainfall, val_df.filter(train_df.columns[columnVals], axis=1)])
    
    X_rain = df_rainfall.to_numpy()
    
    rainfall_1 = np.linalg.norm(X_rain[:, 0:6], ord=1, axis=-1) / 100
    rainfall_2 = np.linalg.norm(X_rain[:, 6:], ord=1, axis=-1) / 100

    # since prev value is found in training set, we don't need to shift by 1
    data = {"rainfall_1": rainfall_1[max_lag:],
            "rainfall_2": rainfall_2[max_lag:]}
        
    for i in range(1, max_lag + 1):
        data["rainfall_2_lag_{}".format(i)] = rainfall_2[max_lag-i:-i]
        data["rainfall_1_lag_{}".format(i)] = rainfall_1[max_lag-i:-i]
        
    hydrometry = pd.concat([train_df["Hydrometry_Nave_di_Rosano"][-1:], val_df["Hydrometry_Nave_di_Rosano"]])
    target = pd.DataFrame({"hydrometry_diff": (hydrometry - hydrometry.shift(1))[1:]}, 
                          index=val_df.index)
    
    exog = pd.DataFrame(data=data, index=target.index)
    
    return exog, target


def get_metrics(pred, gt):
    
    mse = np.power(pred - gt, 2).mean()
    mae = np.abs(pred - gt).mean()
    avg = np.mean(gt)
    r2 = 1 - np.power(pred - gt, 2).sum() / np.power(gt - avg, 2).sum()
    return mse, mae, r2


def get_forecast(exog, target, results, horizon=1, get_level=False): 
    
    pred = []
    gt = []
    level_pred = []
    curr_level_gt = 0 
    T = len(target.index)
    
    # call forecasting method every "horizon" number of steps
    for d in range(0, T, horizon):
        
        test_data = exog.iloc[d:d+horizon] if exog is not None else None
        forecast = results.get_forecast(steps=horizon, exog=test_data) 
    
        # update internal state with ground-truth target 
        results = results.append(target.iloc[d:d+horizon], exog=test_data)
        prediction = forecast.predicted_mean.to_numpy()
        pred.append(prediction)
        
        if get_level:
            # compute cumulative sum of changes over forecasting horizon and add initial level 
            level_pred.append(np.cumsum(prediction) + curr_level_gt)
            curr_level_gt += target.iloc[d:d+horizon].to_numpy().sum()
            
            
    pred = np.array(pred).reshape(-1,)
    gt = target.to_numpy().reshape(-1,)
    
    fig, ax = plt.subplots()
    ax.plot(pred, alpha=0.5, label="forecast")
    ax.plot(gt, alpha=0.5, label="target")
    ax.legend(loc="upper left")
    
    mse, mae, r2 = get_metrics(pred, gt)
    
    print("MSE: {0:.3g}".format(mse))
    print("MAE: {0:.3g}".format(mae))
    print("R2: {0:.3g}".format(r2))
    
    if get_level:
        level_pred = np.array(level_pred).reshape(-1,)
        level_gt = np.cumsum(gt)
    
        mse, mae, r2 = get_metrics(level_pred, level_gt)
        print("Level MSE: {0:.3g}".format(mse))
        print("Level MAE: {0:.3g}".format(mae))
        print("Level R2: {0:.3g}".format(r2))
        
        fig, ax = plt.subplots()
        ax.plot(level_pred, alpha=0.5, label="Level forecast")
        ax.plot(level_gt, alpha=0.5, label="Level target")
        ax.legend(loc="upper left")

    return pred, gt


def wrangle_amiata(filepath):
    
    """
    This function transforms the Amiata spring dataset, making it ready for modelling, then
    cleans and engineers new features for the Amiata dataset
    """
    #Load the dataset into a dataframe
    df = pd.read_csv(filepath)
    
    #Changing the Datetime column from object type
    df.Date = pd.to_datetime(df.Date)

    #Creating more time features
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df['month'] = pd.DatetimeIndex(df['Date']).month
    df['day'] = pd.DatetimeIndex(df['Date']).day
    df['day_of_year'] = pd.DatetimeIndex(df['Date']).dayofyear
    df['week_of_year'] = pd.DatetimeIndex(df['Date']).weekofyear
    df['quarter'] = pd.DatetimeIndex(df['Date']).quarter
    df['season'] = df['month'] % 12 // 3 + 1
    
    #Setting the date column as index
    df = df.set_index('Date')
    
    #Encoding the cyclic feature
    month_in_year = 12
    df['month_sin'] = np.sin(2*np.pi*df['month']/month_in_year)
    df['month_cos'] = np.cos(2*np.pi*df['month']/month_in_year)
    
    #Slice out years with predominantly null values between year 2000 to 2015
    df = df[df['year']>2015]

    #Interpolate to fill NaNs using the default method
    for i in df.columns:
        df[i] = df[i].interpolate()

    #Fill any possible remaining NaN rows with zeros
    df=df.fillna(0)
    
    #Drop multicollinear features
    drop_cols = ['Rainfall_S_Fiora', 'Depth_to_Groundwater_S_Fiora_8', 'Depth_to_Groundwater_S_Fiora_11bis',
                'Temperature_S_Fiora']

    df.drop(drop_cols, inplace=True, axis=1)

    return df

def prep_target_amiata(df, target='Arbure'):

    """
    This function prepares the cleaned Amiata dataset for modelling
    """
    #Isolate the column name as specified in the variable
    target_name = [t for t in df.columns if target in t]
    
    #Initiate a list that wil contain names of relevant columns
    target = [] + target_name
    
    #Iterate through the entire column names and find relevant columns, then append to above list
    for col in df.columns:
        if 'Flow' not in col:
            target.append(col)
    df = df[target]
    
    y = df[target_name].iloc[:,0] #Needed to convert the dataframe target to series
    X = df.drop(target_name, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    
    return X_train, X_test, y_train, y_test

def conditional_impute1(input_df, col, choice='mean'):
    # your code here
    output_df = input_df
    meaner = output_df.groupby(['month','day_of_year'])[col].mean()
    medianer = output_df.groupby(['month','day_of_year'])[col].median()
    def fill_missing_mean(row):
        if pd.isnull(row[col]):
            return meaner[row['month'],row['day_of_year']]
        else:
            return row[col]
    
    def fill_missing_median(row):
        if pd.isnull(row[col]):
            return medianer[row['month'],row['day_of_year']]
        else:
            return row[col]
    
    if choice == 'mean':
        output_df[col] = output_df.apply(fill_missing_mean, axis=1)
    elif choice == 'median':
        output_df[col] = output_df.apply(fill_missing_median, axis=1)
    
    return output_df

def wrangle_auser(filepath):
    
    """
    This function transforms the auser spring dataset, making it ready for modelling, then
    cleans and engineers new features for the Amiata dataset
    """
    #Load the dataset into a dataframe
    df = pd.read_csv(filepath)
    
    #Changing the Datetime column from object type
    df.Date = pd.to_datetime(df.Date)

    #Creating more time features
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df['month'] = pd.DatetimeIndex(df['Date']).month
    df['day'] = pd.DatetimeIndex(df['Date']).day
    df['day_of_year'] = pd.DatetimeIndex(df['Date']).dayofyear
    df['week_of_year'] = pd.DatetimeIndex(df['Date']).weekofyear
    df['quarter'] = pd.DatetimeIndex(df['Date']).quarter
    df['season'] = df['month'] % 12 // 3 + 1
    
    #Setting the date column as index
    df.set_index('Date', inplace = True)
    
    #Slice out years with predominantly null values between year 2000 to 2015
    df = df[df['year']>2015]

    #Drop non-relevant columns
    drop_cols = ['Rainfall_Fabbriche_di_Vallico', 'Rainfall_Calavorno','Rainfall_Piaggione','Rainfall_Borgo_a_Mozzano', 'Temperature_Ponte_a_Moriano', 'Hydrometry_Piaggione', 'day', 'week_of_year']
    df = df.drop(drop_cols, axis = 1)

    return df


def prep_target_auser(df, target='Groundwater_CoS'):

    """
    This function prepares the cleaned Amiata dataset for modelling
    """
    #Isolate the column name as specified in the variable
    target_name = [t for t in df.columns if target in t]
    
    #Initiate a list that wil contain names of relevant columns
    target = [] + target_name
    
    #Iterate through the entire column names and find relevant columns, then append to above list
    for col in df.columns:
        if 'Groundwater' not in col:
            target.append(col)
    df = df[target]
    
    #df.set_index('Date', inplace = True)

    #Drop NaN rows
    df = df.dropna()

    #Split to training and test
    y = df[target_name].iloc[:,0] #Needed to convert the dataframe target to a Series
    X = df.drop(target_name, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
    
    return X_train, X_test, y_train, y_test

def prep_target_bilancino(df, target='Flow_Rate'):

    """
    This function prepares the cleaned Bilancino dataset for modelling
    """
    #Isolate the column name as specified in the variable
    target_name = [t for t in df.columns if target in t]
    
    #Initiate a list that wil contain names of relevant columns
    targets = [] + target_name
    
    #Iterate through the entire column names and find relevant columns, then append to above list
    for col in df.columns:
        if 'Flow_Rate' not in col:
            if 'Lake_Level' not in col:
                targets.append(col)
    df = df[targets]
    
    #df.set_index('Date', inplace = True)

    #Drop NaN rows
    df = df.dropna()

    #Split to training and test
    y = df[target_name].iloc[:,0] #Needed to convert the dataframe target to a Series
    X = df.drop(target_name, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
    
    return X_train, X_test, y_train, y_test

def plot_feature_importance(model):

    feature_names = model.feature_names_in_
    importances = model.feature_importances_
    model_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    model_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()