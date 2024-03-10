import json

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from domus_mlsim.json import lda_from_json, lda_to_json, lr_from_json, lr_to_json


def test_serialization_deserialization():
    # Create example data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 3, 5])

    # Initialize and fit MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)

    # Initialize and fit LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    # Serialize and then deserialize the objects
    json_data = lr_to_json(scaler, model)
    loaded_scaler, loaded_model = lr_from_json(json_data)

    # Assertions to ensure the original and loaded objects are equivalent
    # Test MinMaxScaler
    np.testing.assert_array_almost_equal(scaler.min_, loaded_scaler.min_)
    np.testing.assert_array_almost_equal(scaler.scale_, loaded_scaler.scale_)
    np.testing.assert_array_almost_equal(scaler.data_min_, loaded_scaler.data_min_)
    np.testing.assert_array_almost_equal(scaler.data_max_, loaded_scaler.data_max_)
    np.testing.assert_array_almost_equal(scaler.data_range_, loaded_scaler.data_range_)

    # Test LinearRegression
    np.testing.assert_array_almost_equal(model.coef_, loaded_model.coef_)
    assert model.intercept_ == loaded_model.intercept_


def test_serialization_structure():
    # Create example data
    X = np.array([[7, 8], [9, 10], [11, 12]])
    y = np.array([2, 4, 6])

    # Initialize and fit MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)

    # Initialize and fit LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    # Serialize the objects
    json_data = lr_to_json(scaler, model)

    # Check if the serialization contains the expected keys
    data = json.loads(json_data)
    assert "scaler" in data
    assert "linear_regression" in data
    assert "min_" in data["scaler"]
    assert "scale_" in data["scaler"]
    assert "coef_" in data["linear_regression"]
    assert "intercept_" in data["linear_regression"]


def test_lda_outcomes():
    # Sample data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    # Train StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_X = scaler.transform(X)

    # Train LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver="lsqr")
    lda.fit(scaled_X, y)
    lda_predictions = lda.predict(X)

    # Serialize and Deserialize
    json_data = lda_to_json(scaler, lda)
    deserialized_scaler, deserialized_lda = lda_from_json(json_data)

    # Test for equivalent outcomes
    descaled_X = deserialized_scaler.transform(X)
    np.testing.assert_array_almost_equal(
        scaled_X,
        descaled_X,
        err_msg="Scaled vectors do not match after deserialization.",
    )

    deserialized_lda_predictions = deserialized_lda.predict(descaled_X)
    np.testing.assert_array_equal(
        lda_predictions,
        deserialized_lda_predictions,
        err_msg="LDA predictions do not match after deserialization.",
    )
