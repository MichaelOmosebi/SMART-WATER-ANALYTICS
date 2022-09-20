
# Streamlit dependencies
import streamlit as st
from PIL import Image

st.set_page_config(
	page_title = "ACEA Water Analytics",
	page_icon = "💦"
)

st.title("ACEA SMART WATER ANALYTICS")
st.subheader("Predicting Water Availability ")
st.sidebar.success("# Welcome 🌊")

#Insert an Introduction to the Project; Problem statement, Scope etc...
st.markdown('The Acea Group is one of the leading Italian multiutility operators. Listed on the Italian Stock Exchange since 1999, the company manages and develops water and electricity networks and environmental services. Acea is the foremost Italian operator in the water services sector supplying 9 million inhabitants in Lazio, Tuscany, Umbria, Molise, Campania')

st.markdown('In this project we will focus on the **water sector** to help Acea Group reserve precious water sources.')

st.markdown('Different water bodies, each with unique characteristics, will be explored for this project. Each dataset provided represents a different kind of waterbody. As each waterbody is different from the other, the related features are also different.')

image = Image.open('Water Available.png')
st.image(image, use_column_width=False)