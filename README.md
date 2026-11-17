# ML Linear Regression

A from-scratch linear regression project that estimates a car's price based on its mileage. The model is trained using batch gradient descent and applies Z-score normalization to the mileage feature.

## Features

- Implement one-variable linear regression without machine learning frameworks.
- Normalize mileage data using Z-score normalization.
- Estimate car prices interactively.
- Evaluate model performance using the R-squared metric.
- Visualize regression results, data distributions, box plots, and training convergence.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn

Install the dependencies:

```bash
python3 -m pip install numpy pandas matplotlib seaborn
```

## Dataset

Create a `data.csv` file in the project root with the following columns:

```csv
km,price
240000,3650
139800,3895
```

| Column | Description |
|---|---|
| `km` | Car mileage |
| `price` | Corresponding car price |

Both columns must contain numeric values.

## Usage

Run all commands from the project root.

### 1. Train the Model

```bash
python3 gradient_descent.py
```

Reads `data.csv`, trains the model for 300 iterations, and saves the learned parameters to `values.txt`.

### 2. Estimate a Car Price

```bash
python3 price_estimator.py
```

Enter the car's mileage when prompted to receive an estimated price.

### 3. Evaluate Model Performance

```bash
python3 precision.py
```

Calculates and displays the model's R-squared score as a percentage.

### 4. Visualize Results

```bash
python3 visualize.py
```

Generates visualizations for:

- Mileage vs. price regression.
- Training convergence.
- Mileage and price distributions.
- Mileage and price box plots.

## Project Structure

| File | Description |
|---|---|
| `gradient_descent.py` | Trains the model and saves its parameters. |
| `price_estimator.py` | Estimates car prices based on entered mileage. |
| `precision.py` | Calculates the model's R-squared score. |
| `visualize.py` | Generates model and dataset visualizations. |
| `utils.py` | Provides data loading, normalization, and price calculation utilities. |
| `data.csv` | Contains the training dataset. |
| `values.txt` | Stores the trained model parameters. |

## Notes

- Run `gradient_descent.py` again whenever the dataset changes.
- `price_estimator.py`, `precision.py`, and `visualize.py` require both `data.csv` and the generated `values.txt`.
- Mileage normalization uses the training dataset's mean and standard deviation. The same statistics are reused during prediction to ensure consistency.
