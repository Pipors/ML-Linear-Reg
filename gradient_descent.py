import numpy as np
from pathlib import Path
from utils import read_file, normalize


def gradient_descent(x: np.array, y: np.array, m: int, iter: int):
    """
    Perform gradient descent to optimize the parameters for linear regression.

    Parameters:
    x (np.array): The input feature array.
    y (np.array): The target output array.
    m (int): The number of training examples.
    iter (int): The number of iterations for the gradient descent algorithm.

    Returns:
    tuple: A tuple containing the optimized parameters (theta_0, theta_1).
    """
    learning_rate = 0.1
    theta_0 = 0
    theta_1 = 0
    convergence_points = []
    # convergence_points_1 = []
    for _ in range(iter):
        error = (theta_0 + (theta_1 * x)) - y
        theta_0 -= learning_rate * (1 / m) * error.sum()
        theta_1 -= learning_rate * (1 / m) * (error * x).sum()
        convergence_points.append(abs(int(error.sum())))

    return theta_0, theta_1, convergence_points


def main():
    df = read_file('data.csv')
    # km = normalize(np.array(df['km']))
    km = np.array(df['km'], dtype=float)
    price = np.array(df['price'])

    t0, t1, _ = gradient_descent(x=normalize(km), y=price, m=len(df), iter=300)

    if Path('values.txt').exists():
        with open('values.txt', 'w') as f:
            f.write(f"{t0}\n{t1}\n")
    else:
        with open('values.txt', 'x') as f:
            f.write(f"{int(t0)}\n{int(t1)}\n")
    print("\x1B[1m\x1B[34m◎ File values.txt"
          "has been created successfully ◎\x1B[0m")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\x1B[31m{e}")
