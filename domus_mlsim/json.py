import json
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def lr_to_json(scaler: MinMaxScaler, model: LinearRegression) -> str:
    """Serialize scaler and LR model to JSON"""
    data: Dict[str, Any] = {
        "scaler": {
            "min_": scaler.min_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "data_min_": scaler.data_min_.tolist(),
            "data_max_": scaler.data_max_.tolist(),
            "data_range_": scaler.data_range_.tolist(),
        },
        "linear_regression": {
            "coef_": model.coef_.tolist(),
            "intercept_": model.intercept_.tolist(),
        },
    }
    return json.dumps(data)


def lr_to_json_file(scaler: MinMaxScaler, model: LinearRegression, fname: str) -> None:
    json_data = lr_to_json(scaler, model)

    with open(fname, "w") as f:
        f.write(json_data)


def lr_from_json(json_data: str) -> Tuple[MinMaxScaler, LinearRegression]:
    """Function to deserialize scaler and LR model from JSON"""
    data: Dict[str, Any] = json.loads(json_data)

    # Create new instances
    new_scaler = MinMaxScaler()
    new_model = LinearRegression()

    # Manually set the attributes for MinMaxScaler
    new_scaler.min_ = np.array(data["scaler"]["min_"])
    new_scaler.scale_ = np.array(data["scaler"]["scale_"])
    new_scaler.data_min_ = np.array(data["scaler"]["data_min_"])
    new_scaler.data_max_ = np.array(data["scaler"]["data_max_"])
    new_scaler.data_range_ = np.array(data["scaler"]["data_range_"])
    new_scaler.n_features_in_ = len(new_scaler.scale_)  # Important for sklearn 0.24+

    # Manually set the attributes for LinearRegression
    new_model.coef_ = np.array(data["linear_regression"]["coef_"])
    new_model.intercept_ = data["linear_regression"]["intercept_"]

    return new_scaler, new_model


def lr_from_json_file(fname: str) -> Tuple[MinMaxScaler, LinearRegression]:
    with open(fname, "r") as f:
        loaded_json_data = f.read()

    return lr_from_json(loaded_json_data)


def lda_to_json(scaler: StandardScaler, lda: LinearDiscriminantAnalysis) -> str:
    data: Dict[str, Any] = {
        "standard_scaler": {
            "mean_": scaler.mean_.tolist(),
            "var_": scaler.var_.tolist(),
            "scale_": scaler.scale_.tolist(),
        },
        "lda": {
            "solver": lda.solver,
            "shrinkage": lda.shrinkage,
            "priors": lda.priors.tolist() if lda.priors is not None else None,
            "n_components": lda.n_components,
            "store_covariance": lda.store_covariance,
            "tol": lda.tol,
            "classes_": lda.classes_.tolist(),
            "priors_": lda.priors_.tolist(),
            "_max_components": lda._max_components,
            "means_": lda.means_.tolist(),
            "covariance_": lda.covariance_.tolist(),
            "coef_": lda.coef_.tolist(),
            "intercept_": lda.intercept_.tolist(),
        },
    }
    return json.dumps(data)


def lda_from_json(json_data: str) -> Tuple[StandardScaler, LinearDiscriminantAnalysis]:
    data = json.loads(json_data)

    scaler = StandardScaler()
    scaler.mean_ = np.array(data["standard_scaler"]["mean_"])
    scaler.var_ = np.array(data["standard_scaler"]["var_"])
    scaler.scale_ = np.array(data["standard_scaler"]["scale_"])
    scaler.n_features_in_ = len(scaler.mean_)

    lda_data = data["lda"]
    lda = LinearDiscriminantAnalysis(
        solver=lda_data["solver"],
        shrinkage=lda_data["shrinkage"],
        priors=np.array(lda_data["priors"]) if lda_data["priors"] is not None else None,
        n_components=lda_data["n_components"],
        store_covariance=lda_data["store_covariance"],
        tol=lda_data["tol"],
    )

    # Set attributes that are determined after fitting
    lda.classes_ = np.array(lda_data["classes_"])
    lda.priors_ = np.array(lda_data["priors_"])
    lda.means_ = np.array(lda_data["means_"])
    lda.covariance_ = np.array(lda_data["covariance_"])
    lda.coef_ = np.array(lda_data["coef_"])
    lda.intercept_ = np.array(lda_data["intercept_"])
    lda._max_components = lda_data["_max_components"]

    return scaler, lda


def lda_to_json_file(
    scaler: StandardScaler, lda: LinearDiscriminantAnalysis, filename: str
):
    json_data = lda_to_json(scaler, lda)
    with open(filename, "w") as f:
        f.write(json_data)


def lda_from_json_file(
    filename: str,
) -> Tuple[StandardScaler, LinearDiscriminantAnalysis]:
    with open(filename, "r") as f:
        json_data = f.read()
    return lda_from_json(json_data)
