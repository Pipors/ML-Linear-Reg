+# ML Linear Regression
+
+A from-scratch linear regression project that estimates a car's price from its mileage. The model is trained with batch gradient descent and uses z-score normalization for the mileage feature.
+
+## Features
+
+- Train a one-variable linear regression model without a machine-learning framework.
+- Normalize mileage data before training and prediction.
+- Estimate a car price interactively.
+- Calculate the model's R-squared precision.
+- Visualize the regression line, distributions, box plots, and training convergence.
+
+## Requirements
+
+- Python 3
+- NumPy
+- pandas
+- Matplotlib
+- seaborn
+
+Install the dependencies with:
+
+```bash
+python3 -m pip install numpy pandas matplotlib seaborn
+```
+
+## Dataset
+
+Create a `data.csv` file in the project root with the following columns:
+
+```csv
+km,price
+240000,3650
+139800,3895
+```
+
+The `km` column contains mileage and `price` contains the corresponding car price. Both columns should contain numeric values.
+
+## Usage
+
+Run the commands from the project root.
+
+### 1. Train the model
+
+```bash
+python3 gradient_descent.py
+```
+
+This reads `data.csv`, trains the model for 300 iterations, and writes the learned parameters to `values.txt`.
+
+### 2. Estimate a car price
+
+After training, run:
+
+```bash
+python3 price_estimator.py
+```
+
+Enter the car's mileage when prompted. The script prints the estimated price.
+
+### 3. Check model precision
+
+```bash
+python3 precision.py
+```
+
+This calculates and prints the model's R-squared score as a percentage.
+
+### 4. Display visualizations
+
+```bash
+python3 visualize.py
+```
+
+This opens charts for:
+
+- The mileage/price regression
+- Training convergence
+- Mileage and price distributions
+- Mileage and price box plots
+
+## Project structure
+
+| File | Description |
+| --- | --- |
+| `gradient_descent.py` | Trains the model and saves its parameters |
+| `price_estimator.py` | Estimates a price for entered mileage |
+| `precision.py` | Calculates the model's R-squared score |
+| `visualize.py` | Displays model and dataset visualizations |
+| `utils.py` | Data loading, normalization, and price calculation helpers |
+
+## Notes
+
+- Run `gradient_descent.py` whenever the dataset changes.
+- `price_estimator.py`, `precision.py`, and `visualize.py` require both `data.csv` and the generated `values.txt`.
+- Mileage is normalized using the dataset mean and standard deviation, so predictions use the same dataset statistics.
