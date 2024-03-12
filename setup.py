from setuptools import setup, find_packages

setup(
    name="domus_mlsim",
    version="0.2",
    packages=find_packages(),
    install_requires=["simple-pid", "numpy", "pandas", "scikit-learn"],
    package_data={
        "domus_mlsim": ["model/*.json", "model/scenarios.csv"],
    },
)
