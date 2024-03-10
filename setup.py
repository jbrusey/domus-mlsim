from pathlib import Path

from setuptools import setup

model_root = Path("domus_mlsim/model")
model_files = [
    str(p)
    for p in list(model_root.glob("*lr.json"))
    + list(model_root.glob("*.pickle"))
    + list(model_root.glob("scenarios.csv"))
]

setup(
    name="domus_mlsim",
    version="0.1",
    install_requires=["simple-pid", "numpy", "pandas", "scikit-learn"],
    data_files=[("model", model_files)],
)
