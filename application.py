import streamlit as st
from updated_main import predict, residuals, y_train, r2_train  # Ensure these are correctly imported
import pandas as pd
import matplotlib.pyplot as plt

# Function to add background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.govexec.com/media/featured/wwt6.gif");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True  # Corrected line position
    )

# Add the background image
add_bg_from_url()

st.markdown(
    """
    <div style="text-align: center; margin-top: 5px;">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTDQaB3z7wmDwQmROUxci8vNh8_jrPNTrkgHAS2Yx3c5Q&s" alt="Logo" style="width: 300px; margin-bottom: 5px;">
    </div>""", unsafe_allow_html=True
)

# Setup the layout
st.markdown("<h1 style='color: black;'>Garment Production Prediction</h1>", unsafe_allow_html=True)

# Input widgets for user input
department = st.selectbox('Department', options=['Gloves', 'T-Shirt', 'Sweatshirt'])
quarter = st.selectbox('Quarter', options=['Quarter1', 'Quarter2', 'Quarter3', 'Quarter4'])
no_of_workers = st.number_input('Number of Workers', min_value=25, max_value=100, value=50)
defects_day = st.number_input('Unproductive days per month', min_value=1, max_value=10, value=3)

# Predict button
if st.button('Predict'):
    # Prepare the input data
    input_data = {
        'department': department,
        'quarter': quarter,
        'no_of_workers': no_of_workers,
        'defects_day': defects_day,
    }

    # Call the predict function
    prediction = predict(input_data)
    st.write(f"Predicted Productivity: {prediction}")

    # Generate PRE Plot
    fig, ax = plt.subplots()
    ax.scatter(y_train, residuals)
    ax.axhline(lw=2, color='black')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    st.pyplot(fig)

    # Display R-squared value
    st.write(f"R-squared: {r2_train}")
