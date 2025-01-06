import streamlit as st
import pandas as pd
import numpy as np
import pickle  # to load a saved model
import base64  # to open .gif files in streamlit app

# Cache function for value retrieval
@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict.get(val, 0)


def get_value(val, my_dict):
    return my_dict.get(val, 0)


# Sidebar page selection
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])  # two pages

if app_mode == "Home":
    st.title("LOAN PREDICTION:")
    st.image("loan_image.jpg")
    st.markdown("Dataset:")
    data = pd.read_csv("loan_dataset.csv")
    st.write(data.head())
    st.markdown("Applicant Income VS Loan Amount")
    st.bar_chart(data[["ApplicantIncome", "LoanAmount"]].head(20))

elif app_mode == "Prediction":
    st.image("slider-short-3.jpg")
    st.subheader(
        "YOU need to fill all necessary information in order to get a reply to your loan request!"
    )
    st.sidebar.header("Information about the client:")

    # Dictionaries
    gender_dict = {"Male": 1, "Female": 2}
    feature_dict = {"No": 1, "Yes": 2}
    edu = {"Graduate": 1, "Not Graduate": 2}
    prop = {"Rural": 1, "Urban": 2, "Semiurban": 3}

    # Input Features
    ApplicantIncome = st.sidebar.slider("ApplicantIncome", 0, 10000, 0)
    CoapplicantIncome = st.sidebar.slider("CoapplicantIncome", 0, 10000, 0)
    LoanAmount = st.sidebar.slider("LoanAmount in K$", 9.0, 700.0, 200.0)
    Loan_Amount_Term = st.sidebar.selectbox(
        "Loan_Amount_Term", (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0)
    )
    Credit_History = st.sidebar.radio("Credit_History", (0.0, 1.0))
    Gender = st.sidebar.radio("Gender", tuple(gender_dict.keys()))
    Married = st.sidebar.radio("Married", tuple(feature_dict.keys()))
    Self_Employed = st.sidebar.radio("Self Employed", tuple(feature_dict.keys()))
    Dependents = st.sidebar.radio("Dependents", options=["0", "1", "2", "3+"])
    Education = st.sidebar.radio("Education", tuple(edu.keys()))
    Property_Area = st.sidebar.radio("Property_Area", tuple(prop.keys()))

    # Dependents Handling
    dependents_list = [0, 0, 0, 0]
    if Dependents == "0":
        dependents_list[0] = 1
    elif Dependents == "1":
        dependents_list[1] = 1
    elif Dependents == "2":
        dependents_list[2] = 1
    else:
        dependents_list[3] = 1

    # Property Area Handling
    Rural, Urban, Semiurban = 0, 0, 0
    if Property_Area == "Urban":
        Urban = 1
    elif Property_Area == "Semiurban":
        Semiurban = 1
    else:
        Rural = 1

    # Feature Dictionary
    feature_list = [
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        get_value(Gender, gender_dict),
        get_fvalue(Married),
        dependents_list[0],
        dependents_list[1],
        dependents_list[2],
        dependents_list[3],
        get_value(Education, edu),
        get_fvalue(Self_Employed),
        Rural,
        Urban,
        Semiurban,
    ]

    single_sample = np.array(feature_list).reshape(1, -1)

    if st.button("Click to Predict"):
        try:
            # Load GIFs
            with open("6m-rain.gif", "rb") as file_:
                data_url = base64.b64encode(file_.read()).decode("utf-8")

            with open("green-cola-no.gif", "rb") as file:
                data_url_no = base64.b64encode(file.read()).decode("utf-8")

            # Load Model
            loaded_model = pickle.load(open("Random_Forest.sav", "rb"))
            prediction = loaded_model.predict(single_sample)

            if prediction[0] == 0:
                st.error("According to our calculations, you will not get the loan from the bank.")
                st.markdown(
                    f'<img src="data:image/gif;base64,{data_url_no}" alt="No gif">',
                    unsafe_allow_html=True,
                )
            elif prediction[0] == 1:
                st.success("Congratulations!! You will get the loan from the bank.")
                st.markdown(
                    f'<img src="data:image/gif;base64,{data_url}" alt="Yes gif">',
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
