from utils import read_file, normalize
import numpy as np


def r_squared(km, price, theta_0, theta_1) -> float:
    """
    Calculate the R-squared value for a linear regression model.

    Parameters:
    km (array-like): The independent variable (x).
    price (array-like): The dependent variable (y).
    theta_0 (float): The intercept of the linear regression model.
    theta_1 (float): The slope of the linear regression model.

    Returns:
    float: The R-squared value, representing the proportion of variance
        in the dependent variable that is predictable from
        the independent variable.
    """
    y_predicted = theta_0 + (theta_1 * km)
    ss_res = np.sum((price - y_predicted) ** 2)   # your model's error
    ss_tot = np.sum((price - price.mean()) ** 2)  # total variance in data
    return 1 - (ss_res / ss_tot)


def main():
    df = read_file('data.csv')
    with open('values.txt', 'r') as f:
        theta_0 = f.readline()
        theta_1 = f.readline()
        # f.write('\n')
    precision = r_squared(km=normalize(df['km']), price=df['price'],
                          theta_0=int(theta_0), theta_1=int(theta_1)) * 100
    print(f"The precision of the model is: \x1B[1m\x1B[46m{precision:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\x1B[31m{e}")
    except KeyboardInterrupt as e:
        print(f"\x1B[1m\x1B[31m\nProgram unexpectedly stopped. {e}\x1B[0m")
