import streamlit as st 
import pandas as pd 
import joblib 
from main import main_f 





with st.sidebar:
    st.image('img//Heart.jpg')
    st.title('Toxicity Prediction')
    choice = st.radio('Navigation',['HOME', 'TRAIN_YOUR_DATA','TEST_YOUR_DATA','PREDICTION','INSIGHTS','ANALYSIS'])
    st.info('This application will allow you to not only train the model on your own data, but also You can predict the results for your data as well')

# with st.sidebar:
#     st.image('img//Heart.jpg')
#     st.title('hERG Prediction')
#     choice = st.radio('Navigation',['HOME', 'TRAIN_YOUR_DATA','TEST_YOUR_DATA','PREDICTION','INSIGHTS','ANALYSIS'])
#     st.info('This application will not only train the model on your own data, but also You can predict the results for your data as well')

# if choice == 'HOME':
#     st.title('Toxicity Prediction Model')
#     st.markdown("""The human ether-a-go-go-related gene (hERG) potassium channel plays a crucial role in cardiac repolarization. Inhibition of hERG by potential drugs can lead to 
#                 prolongation of the QT interval on the electrocardiogram (ECG), increasing the risk of arrhythmias and sudden death. Therefore, accurate prediction of hERG 
#                 inhibition is essential for drug safety assessment. Quantitative Structure-Activity Relationship (QSAR) models offer a valuable tool for this purpose, by 
#                 correlating molecular descriptors with hERG inhibitory activity""")

if choice == 'HOME':
    st.title(' Accelerate Drug Discovery with AI-Powered Toxicity Prediction')
    st.markdown("""
**Welcome to the future of drug development!** This interactive dashboard empowers you to quickly and reliably assess the potential toxicity of molecules, paving the way for safer, more efficient drug discovery.

**Benefits:**

* **Save time and resources:** Eliminate the need for lengthy, expensive animal testing.
* **Reduce ethical concerns:** Prioritize promising candidates with lower predicted toxicity.
* **Make informed decisions:** Gain valuable insights into the safety profile of your molecules.

**Get Started:**

* **Upload your own molecules:** Simply drag and drop files or paste SMILES strings.
* **Browse curated datasets:** Explore diverse chemical libraries for inspiration.
* **Interact with predictions:** Visualize results, filter by toxicity type, and download data.

**Target Audience:**

* **Scientists:** Deep dive into advanced models and detailed results.
* **Researchers:** Streamline your toxicity assessment workflow.
* **Students:** Learn about the power of AI in drug discovery.

**About the App:**

* Built on trusted datasets and cutting-edge algorithms.
* Delivers accurate and reliable predictions.
* Committed to responsible and ethical use.

**Disclaimer:** This app is for research and exploration purposes only. Consult with experts and regulatory bodies before making drug development decisions.

**Explore the dashboard now and unlock the potential of AI-powered toxicity prediction!**
""")
    
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

if choice == 'TEST_YOUR_DATA':
    st.title('Upload Your Data for Training And Inferential')

    tab1, tab2 ,tab3 = st.tabs(['Home:clipboard',' : weight_lifter','loacal_performance  : cycler :'])

    with tab1:
        st.header("jhfjhfjsfjs")
        st.write("HEllo")
    file = st.file_uploader('Upload your Dataset here')
    if file:
        df = pd.read_csv(file)
        st.dataframe(df)
        metrics = main_f(df)
        st.table(metrics)

if choice == 'PREDICTION':
    st.title('Upload Your Data for Training And Inferential')
    st.title("hERG :red[Prediction] :bar_chart: :chart_with_upwards_trend: :coffee:")
    st.markdown("")


    tab1, tab2 ,tab3 = st.tabs(['Home:clipboard',' : weight_lifter','loacal_performance  : cycler :'])

    with tab1:
        st.header("jhfjhfjsfjs")
        st.write("HEllo")
        file = st.file_uploader('Upload your Dataset here')
        if file:
            df = pd.read_csv(file)
            st.dataframe(df)
            metrics = main_f(df)
            st.table(metrics)






if choice == 'ANALYSIS':

    st.title("hERG :red[Prediction] :bar_chart: :chart_with_upwards_trend: :coffee:")
    st.markdown("")


    tab1, tab2 ,tab3 = st.tabs(['Data:clipboard','global : weight_lifter','loacal_performance  : cycler :'])

    with tab1:
        st.header("jhfjhfjsfjs")
        st.write("HEllo")