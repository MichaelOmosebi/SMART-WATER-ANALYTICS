import streamlit as st

# Data dependencies
import pandas as pd
import numpy as np
import datetime as dt

#Modelling Libraries
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import plot_importance

import joblib,os #To interact with the local drive

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Created module for this Project
import Module_ACEA as MA


#---------------------------------------------------------------------------------------------------------#
# Importing the Data and Creating the ACEA_App
#---------------------------------------------------------------------------------------------------------#


# Wrangle the raw data using the defined functions
Amiata = MA.wrangle_amiata("../Datasets/Water_Spring_Amiata.csv")
Arno = MA.arno_wrangle("../Datasets/River_Arno.csv")
Bilancino = pd.read_csv("../Datasets/Lake_Bilancino.csv").dropna()
Ausers = MA.wrangle_auser("../Datasets/Aquifer_Auser.csv")


options = ["Amiata(Spring)", "Arno (River)", "Bilancino (Lake)", "Auser (Aquifer)"]
selection = st.sidebar.selectbox("Select Water Body to Explore", options)

#Showing Model working for the AUSER Aquifers dataset
if selection == 'Auser (Aquifer)':
    st.sidebar.success("### Predicting Groundwater level in the Auser Aquifers")

    st.write('## Dataset After Feature Engineering')
    
    #Fill the few NaN rows is some columns
    MA.conditional_impute1(Ausers, 'Hydrometry_Monte_S_Quirico', choice='mean')

    st.dataframe(Ausers.head(10))

    st.info('Preparing the Dataset for Prediction Using the Trained Model')
    
    option = st.selectbox(
     'Which Auser Aquifer Target Would You Like to Predict?',
     ('Groundwater_SAL', 'Groundwater_CoS', 'Groundwater_LT2'))
    
    st.write(f'You are running a prediction for {option}🎚️')

    if st.button("➡️ Test Prediction"):

        #Run a Prediction for the Arbure Water body 'Flow rate'
        X_test = MA.prep_target_auser(Ausers, target=option)[1]
        y_test = MA.prep_target_auser(Ausers, target=option)[3]

        #Loading the model
        SAL_model = joblib.load(open(os.path.join("../Datasets/XGBoost_model_auser_SAL.pkl"),"rb"))
        CoS_model = joblib.load(open(os.path.join("../Datasets/XGBoost_model_auser_CoS.pkl"),"rb"))
        LT2_model = joblib.load(open(os.path.join("../Datasets/XGBoost_model_auser_LT2.pkl"),"rb"))
        models = {'Groundwater_SAL': SAL_model,
                'Groundwater_CoS': CoS_model, 
                'Groundwater_LT2': LT2_model}
        predictor = models.get(option)
        #>>>Running predictions>>>
        prediction = predictor.predict(X_test)

        #>>>Creating the plot>>>
        #Creating a plot that compares actual y_val with Randomforest Predicted values
        fig = plt.figure(figsize =(15, 9))
        plt.plot(np.arange(len(y_test)), y_test, color='darkblue', label = 'Actual Flowrate')
        plt.plot(np.arange(len(y_test)), prediction, color="red", label = 'Model Predictions')
        plt.title("Recorded Flowrate Vs Predicted Flowrate values over time", fontsize=18)
        plt.xlabel("time(Days)")
        plt.ylabel(f"upper({option})Flowrate(Litres per second)")
        plt.legend(loc='upper right')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        #Model Performance
        MSE = mean_squared_error(prediction, y_test)
        R2 = r2_score(prediction, y_test)
        MAE = mean_absolute_error(prediction, y_test)

        st.info('How the Model Performed')

        st.write(f'###### R^2 = {round(R2*100, 1)}%')
        st.write(f'###### MAE = {round(MAE,4)} meters from groundfloor')
        st.write(f'###### MSE = {round(MSE, 4)} meters from groundfloor')

        st.info('Identifying the Most Important Features in the Model')
        #Plot Feature Importance
        plot_importance(predictor)
        st.pyplot()

#Showing Model working for the Spring AMIATA dataset
elif selection == 'Amiata(Spring)':
    st.sidebar.success("### Predicting Amiata Spring Flowrate")

    st.write('## Dataset After Feature Engineering')
    
    st.dataframe(Amiata.head(10))

    st.info('Preparing the Dataset for Prediction Using the Trained Model')
    
    option = st.selectbox(
     'Which Amiata Spring Target Would You Like to Predict?',
     ('Arbure', 'Bugnano', 'Ermicciolo'))
    
    st.write(f'You are running a prediction for {option} River 🌊')

    if st.button("➡️ Test Prediction"):

        #Run a Prediction for the Arbure Water body 'Flow rate'
        X_test = MA.prep_target_amiata(Amiata, target=option)[1]
        y_test = MA.prep_target_amiata(Amiata, target=option)[3]

        #Loading the model
        Arbure_model = joblib.load(open(os.path.join("../Datasets/forest_model_Amiata_Arbure.pkl"),"rb"))
        Bugnano_model = joblib.load(open(os.path.join("../Datasets/forest_model_Amiata_Bugnano.pkl"),"rb"))
        Ermicciolo_model = joblib.load(open(os.path.join("../Datasets/forest_model_Amiata_Ermicciolo.pkl"),"rb"))
        models = {'Arbure': Arbure_model,
                'Bugnano': Bugnano_model, 
                'Ermicciolo': Ermicciolo_model}
        predictor = models.get(option)
        #>>>Running predictions>>>
        prediction = predictor.predict(X_test)

        #>>>Creating the plot>>>
        #Creating a plot that compares actual y_val with Randomforest Predicted values
        fig = plt.figure(figsize =(15, 9))
        plt.plot(np.arange(len(y_test)), y_test, color='darkblue', label = 'Actual Flowrate')
        plt.plot(np.arange(len(y_test)), prediction, color="red", label = 'Random Forest Predictions')
        plt.title("Recorded Flowrate Vs Predicted Flowrate values over time", fontsize=18)
        plt.xlabel("time(Days)")
        plt.ylabel(f"upper({option})Flowrate(Litres per second)")
        plt.legend(loc='upper right')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        #Model Performance
        MSE = mean_squared_error(prediction, y_test)
        R2 = r2_score(prediction, y_test)
        MAE = mean_absolute_error(prediction, y_test)

        st.info('How the Model Performed')

        st.write(f'###### R^2 = {round(R2*100, 1)}%')
        st.write(f'###### MAE = {round(MAE,4)} Litres per second')
        st.write(f'###### MSE = {round(MSE, 4)} Litres per second')


        st.info('Identifying the Most Important Features in the Model')
        #Using the defined function in the module
        MA.plot_feature_importance(predictor)

#Showing Model working for the River ARNO dataset
elif selection == 'Arno (River)':
    st.sidebar.success("### Predicting River Arno Water Level Feature")

    st.write('## Dataset After Feature Engineering')
    
    st.dataframe(Arno.head(10))
    if st.button("Test Prediction"):

        st.info('>>>Spinning the model >>> running your predictions')
        
        #Here we use a test_size of 30% --- No validation
        train, val, test = MA.split(Arno, train_frac=0.7, val_frac=0.30)

        #Running a Test Prediction
        exog, target = MA.get_train_features(train, max_lag=1)

        model = ARIMA(endog=target, exog=exog, order=(0, 0, 0))
        results = model.fit()

        exog, target = MA.get_val_features(train, val, max_lag=1)

        pred, gt = MA.get_forecast(exog, target, results, horizon=1, get_level=False)

        prediction = pd.Series(pred)
        y_test = pd.Series(gt)

        fig = plt.figure(figsize =(15, 9))
        plt.plot(np.arange(len(y_test)), y_test, color='darkblue', label = 'Actual Level')
        plt.plot(np.arange(len(y_test)), prediction, color="red", label = 'ARIMA Predictions')
        plt.title("Recorded Level & Predicted Level over time", fontsize=18)

        plt.xlabel("time(Days)")
        plt.ylabel("River Level")
        plt.legend(loc='upper right')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.info('How the Autoregressive Model Performed')
        mse, mae, r2 = MA.get_metrics(prediction, y_test)

        st.write(f'###### R^2 = {round(r2*100, 1)}%')
        st.write(f'###### MAE = {round(mae,4)} meters above river floor')
        st.write(f'###### MSE = {round(mse, 4)} meters above river floor')


#Showing Model working for the BILANCIO Lake dataset
elif selection == 'Bilancino (Lake)':
    st.sidebar.success("### Predicting Lake Bilancino Water Level & Flow Rate features")

    st.write('## Dataset After Feature Engineering')
    Bilancino = Bilancino.dropna()
    Bilancino.Date = pd.to_datetime(Bilancino.Date)

    #Creating new date features
    Bilancino['year'] = pd.DatetimeIndex(Bilancino['Date']).year
    Bilancino['month'] = pd.DatetimeIndex(Bilancino['Date']).month
    Bilancino['day'] = pd.DatetimeIndex(Bilancino['Date']).day
    Bilancino['day_of_year'] = pd.DatetimeIndex(Bilancino['Date']).dayofyear
    Bilancino['week_of_year'] = pd.DatetimeIndex(Bilancino['Date']).weekofyear
    Bilancino['quarter'] = pd.DatetimeIndex(Bilancino['Date']).quarter

    Bilancino.set_index('Date', inplace = True)
    st.dataframe(Bilancino.head(10))

    #Running a Test Prediction
    
    #Clean the df and reset index
    option = st.selectbox(
     'Which of Lake Bilancino Hydrometry Feature Would You Like to Predict?',
     ('Flow_Rate', 'Lake_Level'))
    
    st.write(f'You are running a prediction for Lake Bilancino {option}')

    if st.button("➡️ Test Prediction"):

        #Run a Prediction for the Arbure Water body 'Flow rate'
        X_test = MA.prep_target_bilancino(Bilancino, target=option)[1]
        y_test = MA.prep_target_bilancino(Bilancino, target=option)[3]

        #Loading the model
        FlowRate_model = joblib.load(open(os.path.join("../Datasets/XG_model_Bilancino_Flowrate.pkl"),"rb"))
        LakeLevel_model = joblib.load(open(os.path.join("../Datasets/XG_model_Bilancino_level.pkl"),"rb"))
        models = {'Flow_Rate': FlowRate_model,
                'Lake_Level': LakeLevel_model}
        predictor = models.get(option)
        #>>>Running predictions>>>
        prediction = predictor.predict(X_test)

        #>>>Creating the plot>>>
        #Creating a plot that compares actual y_val with Randomforest Predicted values
        fig = plt.figure(figsize =(15, 9))
        plt.plot(np.arange(len(y_test)), y_test, color='darkblue', label = 'Actual Flowrate')
        plt.plot(np.arange(len(y_test)), prediction, color="red", label = 'Random Forest Predictions')
        plt.title(f"Recorded {option} Vs Predicted {option} over time", fontsize=18)
        plt.xlabel("time(Days)")
        plt.ylabel(f"upper({option})")
        plt.legend(loc='upper right')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        #Model Performance
        MSE = mean_squared_error(prediction, y_test)
        R2 = r2_score(prediction, y_test)
        MAE = mean_absolute_error(prediction, y_test)

        st.info('How the Model Performed')

        st.write(f'###### R^2 = {round(R2*100, 1)}%')
        st.write(f'###### MAE = {round(MAE,4)}')
        st.write(f'###### MSE = {round(MSE, 4)}')

        st.info('Identifying the Most Important Features in the Model')
        #Plot Feature Importance
        plot_importance(predictor)
        st.pyplot()

        #image = Image.open('../Datasets/Lake result.png')
        #st.image(image, use_column_width=False)