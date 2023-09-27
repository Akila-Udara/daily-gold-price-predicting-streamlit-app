# - Importing the dependencies
import pandas as pd
import xgboost as xgb
import joblib
import streamlit as st

# - CSS Styling
st.markdown(
    """
    <style>
    .centered-heading {
        text-align: center;
        padding-bottom: 40px;
    }
    .result {
        text-align:center;
        margin-top: 30px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# - Function to load the model
st.cache_data()
def load_model():
    xgb_model = joblib.load('xgboost_model_daily_gold_price.pkl')
    return xgb_model

# - Loading the XGBoost model
xgb_model = load_model()

# - Function to predict the gold price
def predict_gold_price(features):
    prediction = xgb_model.predict(features)
    return prediction[0]

# - Streamlit App Name
st.markdown("<h1 class='centered-heading'>Gold Price Prediction App</h1>", unsafe_allow_html=True)

# - Image path
image_path = 'gold.png'

# - Centering the image
cola, colb, colc = st.columns(3)

with cola:
    pass

with colb:
    st.image(image_path, width=250)

with colc:
    pass


# - Creating two columns
col1, col2 = st.columns(2)

# - Column 1
with col1:
    # - Year dropdown list (Default - 2023)
    Year = st.selectbox('Year', list(range(2008, 2028)), index=15)
    
# - Column 2
with col2:
    # - Month dropdown list (Default - 1)
    Month = st.selectbox('Month', list(range(1, 13)), index=0)

# - Date dropdown list (Default - 1)
Day = st.selectbox('Date', list(range(1, 32)), index=0)

# - Creating a DataFrame with user inputs
user_input = pd.DataFrame({
    'Day': [Day],
    'Month': [Month],
    'Year': [Year]
})

# - Prediction button
if st.button('Predict'):
    prediction = predict_gold_price(user_input)
    st.markdown(
                f"<h5 class='result'>Predicted Gold Price for {Day}/{Month}/{Year} - <span style='color:green;'>$ {prediction:.2f}</span></h1>",
                unsafe_allow_html=True
            )


            
            
