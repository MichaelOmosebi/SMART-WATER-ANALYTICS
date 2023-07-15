import streamlit as st

# Data dependencies
import pandas as pd
import numpy as np
import datetime as dt
import joblib,os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load your raw data
Amiata_raw = pd.read_csv("Datasets/Water_Spring_Amiata.csv")
Amiata = pd.read_excel("Datasets/Amiata_wrangled.xlsx")
Arno = pd.read_csv("Datasets/River_Arno.csv")
Bilancino = pd.read_csv("Datasets/Lake_Bilancino.csv")
Ausers = pd.read_csv("Datasets/Aquifer_Auser.csv")


options = ["Amiata(Spring)", "Arno (River)", "Bilancino (Lake)", "Auser (Aquifer)"]
selection = st.sidebar.selectbox("Select Water Body to Explore", options)

if selection == 'Auser (Aquifer)':
    st.sidebar.success("### Groundwater --- Like Oasis in the Flourising Tuscany")
    st.write("### The Auser Aquifers Dataset")
    st.write(Ausers.tail())

    st.write("This water body consists of two subsystems, that we call NORTH and SOUTH, where the former partly influences the behaviour of the latter. The levels of the NORTH sector are represented by the values of the SAL, PAG, CoS and DIEC wells, while the levels of the SOUTH sector by the LT2 well.")

    start=Ausers.Date[0]
    end=Ausers.Date[len(Ausers)-1]
    Ausers['Date'] = pd.to_datetime(Ausers['Date'])
    years=len(Ausers['Date'].dt.year.unique())
    st.write(f'The dataset contains observations taken from **{start}** to **{end}** covering **{years}** years.')

    #Checking the dataset shape
    st.write(f"Number of rows is = {Ausers.shape[0]}")
    st.write(f"Number of columns is = {Ausers.shape[1]}")

    st.write('#### Missingness Analysis')
    #An overview of missing values per feature
    st.write('All the features are affected by a huge proportion of missingness, except the Date(Unique key) and Temperature columns')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(rc={'figure.figsize':(11,8)})
    sns.heatmap(Ausers.isnull(),yticklabels=False,cbar=False,cmap="copper")
    st.pyplot()


elif selection == 'Amiata(Spring)':
    st.sidebar.success("### The Hot Springs of Mt. Amiata♨️")
    st.write("## The Amiata Spring Dataset")
    st.write(Amiata_raw.tail())
    st.write("This aquifer is accessed through the **Ermicciolo**, **Arbure**, **Bugnano** and **Galleria Alta** springs. The levels and volumes of the four springs are influenced by the parameters: _pluviometry_, _sub-gradation_, _hydrometry_, _temperatures_ and drainage volumes.")
    start=Amiata_raw.Date[0]
    end=Amiata_raw.Date[len(Amiata_raw)-1]
    Amiata_raw['Date'] = pd.to_datetime(Amiata_raw['Date'])
    years=len(Amiata_raw['Date'].dt.year.unique())
    st.write(f'The dataset contains observations taken between **{start}** and **{end}**, covering **{years}** years.')

    #Checking the dataset shape
    st.write(f"Number of rows is = {Amiata_raw.shape[0]}")
    st.write(f"Number of columns is = {Amiata_raw.shape[1]}")

    st.write('#### Missingness Analysis')
    #An overview of missing values per feature
    st.write('All the features are affected by a significant proportion of missingness, except the Date and Temperature columns')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(rc={'figure.figsize':(11,8)})
    sns.heatmap(Amiata_raw.isnull(),yticklabels=False,cbar=False,cmap="copper")
    st.pyplot()

    st.write()

    #st.write(Amiata_raw.info().to_frame())

elif selection == 'Arno (River)':
    st.sidebar.success("### The Principal stream of the Tuscany Region")
    st.write("### The Arno River Dataset")
    st.write(Arno.tail())
    st.write("The Arno is the second largest river in peninsular Italy and the main waterway in Tuscany and it  has a relatively torrential regime, due to the nature of the surrounding soils (marl and impermeable clays).")
    start=Arno.Date[0]
    end=Arno.Date[len(Arno)-1]
    Arno['Date'] = pd.to_datetime(Arno['Date'])
    years=len(Arno['Date'].dt.year.unique())
    st.write(f'The dataset contains observations taken from **{start}** to **{end}**, covering **{years}** years.')

    #Checking the dataset shape
    st.write(f"Number of rows is = {Arno.shape[0]}")
    st.write(f"Number of columns is = {Arno.shape[1]}")

    st.write('#### Missingness Analysis')
    #An overview of missing values per feature
    st.write('All the features are affected by a huge proportion of missingness, except the Date and Hydrometry columns')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(rc={'figure.figsize':(11,8)})
    sns.heatmap(Arno.isnull(),yticklabels=False,cbar=False,cmap="copper")
    st.pyplot()



elif selection == 'Bilancino (Lake)':
    st.write("### The Bilancino Lake Dataset")
    st.sidebar.success("### Tuscany's Resourceful artificial basin")
    st.write(Bilancino.tail())
    st.write(" It is an artificial lake in Mugello, in the province of Florence. It has a maximum depth of thirty-one metres and a surface area of 5 square kilometres.")

    start=Bilancino.Date[0]
    end=Bilancino.Date[len(Bilancino)-1]
    Bilancino['Date'] = pd.to_datetime(Bilancino['Date'])
    years=len(Bilancino['Date'].dt.year.unique())
    st.write(f'The dataset contains observations taken from **{start}** to **{end}**, covering **{years}** years.')

    #Checking the dataset shape
    st.write(f"Number of rows is = {Bilancino.shape[0]}")
    st.write(f"Number of columns is = {Bilancino.shape[1]}")

    st.write('#### Missingness Analysis')
    #An overview of missing values per feature
    st.write('Except the Date column which is unique and the two target features, all the other features are almost uniformly affected by a minor proportion of missingness')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(rc={'figure.figsize':(11,8)})
    sns.heatmap(Bilancino.isnull(),yticklabels=False,cbar=False,cmap="copper")
    st.pyplot()
