import pandas as pd
from pathlib import Path


def normalize(km, milage=None):
    """
    Normalize the input data using z-score normalization.

    Returns:
    pandas.Series or numpy.ndarray: The normalize ddata with
    mean 0 and standard deviation 1.
    """
    if milage is not None:
        return ((milage - km.mean()) / km.std())
    return (km - km.mean()) / km.std()


def read_file(filename: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns its contents as a pandas DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"\x1B[1m\x1B[31m★ The file {filename} "
                                "does not exists ★\x1B[0m")
    return pd.read_csv(filename)


def calculate_price(theta_0: float, theta_1: float, milage: float) -> float:
    """
    Calculate the price based on the linear regression parameters.

    Returns:
        float: The calculated price based on the provided parameters.
    """
    predicted_price = theta_0 + (theta_1 * milage)
    return max(0, predicted_price)
