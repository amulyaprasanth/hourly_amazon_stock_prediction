import os
import hopsworks
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Get the environment variables
hopsworks_api_key = str(os.getenv("HOPSWORKS_API_KEY"))

# create preprocessor object