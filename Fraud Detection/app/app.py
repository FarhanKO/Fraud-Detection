import sys
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import RobustScaler

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # lets `import src.xxx` resolve regardless of cwd
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))   # lets `from assets... import ...` resolve
 
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
 
from assets.sample_transactions import MOCK_LEGIT, MOCK_FRAUD, RAW_COLS  # noqa: E402
