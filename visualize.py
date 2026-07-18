import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_file, normalize
from gradient_descent import gradient_descent
import pandas as pd


# def visualize_bar(df: pd.DataFrame, x: str, ax: plt.axes, label: str):
#     """
#     Visualizes a bar plot using seaborn.

#     """
#     sns.histplot(data=df, x=x, bins=24, ax=ax)
#     ax.set_xticks(range(24))
#     ax.set_xlabel('Car Nº')
#     ax.set_ylabel(label)
#     ax.set_title(f"Bar plot of {label}")


def visualize_bar(df: pd.DataFrame, column: str, ax: plt.Axes, label: str):
    """
    Visualizes the distribution of a continuous variable using a histogram.
    """
    sns.histplot(data=df, x=column, bins=8, ax=ax)
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel('Frequency')
    ax.set_title(f"Distribution of {label}")


def visualize_box(df: pd.DataFrame, x, ax: plt.axes, label: str):
    """
    Visualizes a box plot for a specified column in a DataFrame.

    Returns:
    None
    """
    sns.boxplot(data=df, x=x, ax=ax)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, df[x].max(), num=8))
    ax.margins(0.05)
    ax.set_title(f"Box plot of {label}")


# visualizing the scatter plot km/price
def visualize_regplot(df, x, y, ax: plt.axes, a, b):
    """
    Visualizes a regression plot with a fitted line based
    on the provided linear equation.

    Returns:
    None: This function does not return a value but modifies the provided axes.
    """

    # Create line from equation y = ax + b
    x_line = np.linspace(df['km'].min(), df['km'].max(), 100)
    y_line = int(a) * normalize(x_line) + int(b)

    sns.regplot(data=df, x=x, y=y, ax=ax, ci=95)
    ax.set_title('Scatter plot with Regression line')
    ax.set_xlabel('km')
    ax.set_ylabel('price')
    ax.plot(x_line, y_line, color='green', label=f"y = {a}x + {b}")
    plt.margins(x=0.001)


def main():
    df = read_file('data.csv')
    *_, convergence_points = gradient_descent(
        x=normalize(df['km']), y=df['price'], m=len(df), iter=300)
    a, b = 0, 0
    with open('values.txt', 'r') as f:
        b = f.readline()
        a = f.readline()

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)

    # Row 1 (scatter spanning 2 columns)
    reg = fig.add_subplot(gs[0, 0])
    conver = fig.add_subplot(gs[0, 1])

    # Row 2 (two histograms)
    bar_km = fig.add_subplot(gs[1, 0])
    bar_price = fig.add_subplot(gs[1, 1])

    # Row 3 (two boxplots)
    box_km = fig.add_subplot(gs[2, 0])
    box_price = fig.add_subplot(gs[2, 1])

    visualize_regplot(df=df, x='km', y='price', ax=reg, a=a, b=b)
    visualize_bar(df=df, column='km', ax=bar_km, label='km')
    visualize_bar(df=df, column='price', ax=bar_price,  label='price')
    visualize_box(df=df, x='km', ax=box_km, label='km')
    visualize_box(df=df, x='price', ax=box_price, label='price')
    conver.plot(convergence_points)
    conver.set_title('Convergence plot')
    conver.set_ylabel('errors')
    fig.legend(
        bbox_to_anchor=(0.2, 0.94, 0.35, 0.04),
        bbox_transform=fig.transFigure,
        ncols=5,
        fontsize=10,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"\nGracefully stopping program: {e}\nExited")
    except BaseException as e:
        print({e})
