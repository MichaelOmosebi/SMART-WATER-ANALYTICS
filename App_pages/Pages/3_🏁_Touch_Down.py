import streamlit as st
from PIL import Image

st.markdown("# CONCLUSION ")
st.sidebar.markdown("### Data Science Team 🧑🏻‍🤝‍🧑🏿")
st.sidebar.markdown('Titus Wanjohi Olang: tityewanjohi@gmail.com')
st.sidebar.markdown('Ireoluwa Olaiya: olaiyaoreoluwa3@gmail.com')
st.sidebar.markdown('Christian DivineFavour: christiandivinefavour@gmail.com')
st.sidebar.markdown('Michael Kanu:      michaelokanu01@yahoo.com')
st.sidebar.markdown('Gabriel Asiegbu:       gwap2@live.com')
st.sidebar.markdown('Michael Omosebi:   omosebimichael@live.com')

st.info('Our Approach & Scope')
st.write('The scope of this project was to show how viable it is to predict the availability of water from all four natural water body types, namely: ```Springs```, ```Lakes```, ```Rivers```, and ```Aquifers```.')
st.write(' ')
st.write('')

image = Image.open('../Datasets/Project_Flow.gif')
st.image(image, caption='Project Timeline & Approach', use_column_width=True)

st.info('Take Out')
st.write('This project has been able to show that, using quality data, the primary hydrology features for the different water bodies can be predicted, hence predicting water availability.')

st.write(' ')
st.write(' ')
st.write(' ')

st.info('#### Recommendations')
st.write('1) Having identified the most important features, it is important to prioritize such during data gathering for future water projects.')
st.write(' ')
st.write('2) Also, A better understanding of the geaographical features in the terrain could help to make a more effective decision in feaure engineering and feature selection.')