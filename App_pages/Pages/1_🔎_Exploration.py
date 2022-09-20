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
Amiata_raw = pd.read_csv("../Datasets/Water_Spring_Amiata.csv")
Amiata = pd.read_excel("../Datasets/Amiata_wrangled.xlsx")
Arno = pd.read_csv("../Datasets/River_Arno.csv")
Bilancino = pd.read_csv("../Datasets/Lake_Bilancino.csv")
Ausers = pd.read_csv("../Datasets/Aquifer_Auser.csv")


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
    st.pyplot(sns)


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
    sns.set(rc={'figure.figsize':(11,8)})
    sns.heatmap(Bilancino.isnull(),yticklabels=False,cbar=False,cmap="copper")
    st.pyplot()

    

# """
# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 
# 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 
# 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 
# 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 
# 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 
# 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 
# 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 
# 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 
# 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 
# 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 
# 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 
# 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 
# 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 
# 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
# """