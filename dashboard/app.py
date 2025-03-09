import os
import hopsworks
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import altair as alt


hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")

# set page config
st.set_page_config(
    page_title="Amazon Stock Prediction",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")
st.title("💹Amazon Stock Prediction")

## Load data
# Login in to feature store
if "fs" and "mr" not in st.session_state:
    project = hopsworks.login(api_key_value=str(hopsworks_api_key))
    st.session_state.fs = project.get_feature_store()

# Get the data
if "data" not in st.session_state:

    amazon_fv = st.session_state.fs.get_feature_view("amazon_fv", version=1)

    st.session_state.data = amazon_fv.get_batch_data().sort_values("datetime")

# get the predictions data frame as well
if "predictions" not in st.session_state:
    st.session_state.predictions = st.session_state.fs.get_feature_group(
        "amazon_stock_predictions", version=1
    ).read()
    st.session_state.predictions = pd.DataFrame(st.session_state.predictions).sort_values("datetime")


# Function to plot historical data
def plot_historical_data():
    st.subheader("Historical Data")
    st.write(
        px.line(
            st.session_state["data"],
            x="datetime",
            y="close",
            title="Amazon Stock Price",
        )
    )


def plot_candle_chart(stock_data):
    st.subheader("Candlestick Chart")
    candlestick_chart = go.Figure(
        data=[
            go.Candlestick(
                x=stock_data.datetime,
                open=stock_data["open"],
                high=stock_data["high"],
                low=stock_data["low"],
                close=stock_data["close"],
            )
        ]
    )
    candlestick_chart.update_layout(
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(candlestick_chart, use_container_width=True)


# Create the initial plot with historical data
def plot_predictions():
    # Plot close values for last 4 days and plot the predictions as future values
    last_7_days_data = st.session_state.data.tail(49)
    
    # Combine historical data and predictions
    combined_data = pd.concat([last_7_days_data[['datetime', 'close']], st.session_state.predictions[['datetime', 'close']].tail(7)])
    
    # Create the plot with combined data
    fig = px.line(combined_data, x='datetime', y='close', title="Close Values for Last 28 Days and Predictions")
    
    # Add predictions to the plot
    fig.add_trace(
        go.Scatter(
            x=st.session_state.predictions.tail(7)['datetime'],
            y=st.session_state.predictions.tail(7)['close'],
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        )
    )
    
    st.write(fig)

st.subheader("Today's Stock Predictions: ")
cols = st.columns(7)
for i, col in enumerate(cols):
    with col:
        last_value = st.session_state.predictions.iloc[-(i+1)].close
        st.metric(
            f"{st.session_state.predictions.iloc[-(i+1)].id}",
            value=f"${last_value:.2f}",
        )
plot_predictions()
plot_candle_chart(st.session_state["data"])

# Disclaimer
st.markdown("---")
st.markdown(
    """
    **Disclaimer:**  
    This application is for educational purposes only and is not intended to provide financial or investment advice. 
    The stock predictions are based on historical data and algorithms, which may not predict future market behavior. 
    Users should not rely on this application for trading decisions. Always consult with a qualified financial advisor 
    before making any investment choices. The creators are not responsible for any financial losses or damages resulting 
    from the use of this app.
    """
)

