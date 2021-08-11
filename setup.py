from setuptools import setup

from pathlib import Path

model_root = Path("domus_mlsim/model")
model_files = [
    str(p)
    for p in list(model_root.glob("*lr.joblib"))
    + list(model_root.glob("*.pickle"))
    + list(model_root.glob("scenarios.csv"))
]

setup(
    name="domus_mlsim",
    version="0.1",
    install_requires=["simple-pid", "numpy", "pandas", "scikit-learn==0.23.2"],
    data_files=[("model", model_files)],
)
