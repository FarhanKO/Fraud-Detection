import random
import joblib
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def save_model(model, filepath):
    joblib.dump(model, filepath)


def load_model(filepath):
    return joblib.load(filepath)


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def format_percentage(value):
    return f"{value * 100:.2f}%"
