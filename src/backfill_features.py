import os
import requests
import hopsworks
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the environment variables
tiingo_api_key = os.getenv("TIINGO_API_KEY")
hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")