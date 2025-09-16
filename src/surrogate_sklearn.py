# src/surrogate_sklearn.py
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model(hidden=(512,512,256), max_iter=700):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=hidden,
                             activation="relu", solver="adam",
                             max_iter=max_iter, random_state=0))
    ])

def train_and_save(X, Y, out_path="models/mlp_wideband.joblib"):
    model = build_model()
    model.fit(X, Y)
    joblib.dump(model, out_path)
    return out_path

def load_model(path="models/mlp_wideband.joblib"):
    return joblib.load(path)
