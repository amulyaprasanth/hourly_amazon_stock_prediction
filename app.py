import streamlit as st
from PIL import Image

# Title
st.title("Amazon Stock Predictions")

# Disclaimer
st.markdown("""
**Disclaimer:** The predictions provided by this model are for informational and educational purposes only. 
They should not be considered financial advice or used as the sole basis for making investment decisions. 
We do not guarantee the accuracy, completeness, or reliability of the predictions. 
Use at your own risk, and consult a qualified financial professional before making any trades.
""")

# Display images side by side with headings and some gap in the middle
col1, col2, col3 = st.columns([1, 0.1, 1])

with col1:
    st.header("Yesterday's Predictions")
    yesterday_predictions_image = Image.open("assets/yesterdays_predictions.png")
    st.image(yesterday_predictions_image, caption="Amazon Stock Predictions for Yesterday")

with col2:
    st.write("")  # Empty column for gap

with col3:
    st.header("Today's Predictions")
    today_predictions_image = Image.open("assets/todays_predictions.png")
    st.image(today_predictions_image, caption="Amazon Stock Predictions for Today")
