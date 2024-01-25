import streamlit as st
import pandas as pd
from main import main_f


st.title("Welcome To Training And Inference")
st.markdown("This is a training and inference")

choice = st.radio('Navigation',['HOME', 'TRAIN_YOUR_DATA'])


if choice == 'HOME':
    st.title('Home')
    st.header('Home')
    st.markdown('Home')



    
if choice == 'TRAIN_YOUR_DATA':
    st.title('Training And Inference With Your Data')

    tab1, tab2 ,tab3 = st.tabs(['Home:clipboard:','Gloobal :weight_lifter:','local_performance :cycler:'])

    with tab1:
        st.header("Your Data")
        st.write("Upload your dataset. It should contain only two columns as 'SMILES' and 'ACT'")
        file = st.file_uploader('Upload your Dataset here')
        if file:
            df = pd.read_csv(file)
            st.dataframe(df)
            metrics = main_f(df)
            st.table(metrics)