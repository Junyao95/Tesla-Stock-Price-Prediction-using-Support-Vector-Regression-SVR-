# Tesla Stock Price Prediction using Support Vector Regression (SVR)
This project applies Support Vector Regression (SVR) to predict Tesla's stock prices using three different SVR models: Linear, Polynomial, and Radial Basis Function (RBF). The dataset is obtained using the yfinance library to retrieve the past two years of Tesla's stock prices.

## Project Structure
- svr_stock_prediction.py: The Python script that contains the logic to fetch the data, train the SVR models, and visualize the results.
- Requirements: Required Python packages to run the project.

## Prerequisites
You need the following Python libraries to run the code:
- numpy
- matplotlib
- scikit-learn
- yfinance
You can install these libraries using the following commands:
```
bash

pip install numpy matplotlib scikit-learn yfinance
```
## Code Overview

### Step 1: Load Tesla Stock Data from Yahoo Finance
The script uses the yfinance library to fetch Tesla's historical stock prices for the past two years. The closing price is used as the target value for the SVR models.
```
python

import yfinance as yf
tsla = yf.Ticker("TSLA")
tsla_data = tsla.history(period="2y")
```
### Step 2: Prepare Data for the Models
The dates are represented as numerical indices (np.arange(len(tsla_data))), and the closing prices are extracted. The dates are reshaped to match the input format required by the SVR models.

### Step 3: Train Three SVR Models
Three different SVR models are trained on the data:
1. Linear SVR: Uses a linear kernel.
2. Polynomial SVR: Uses a polynomial kernel (degree 2).
3. RBF SVR: Uses the Radial Basis Function (RBF) kernel.
```
python 

svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

svr_lin.fit(dates, prices)
svr_poly.fit(dates, prices)
svr_rbf.fit(dates, prices)
```
### Step 4: Plot and Compare the Models
The script uses matplotlib to plot the actual stock prices alongside the predictions from the three SVR models (Linear, Polynomial, and RBF).
```
python

plt.scatter(dates, prices, color='black', label='Data')
plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
plt.legend()
plt.show()
```
### Step 5: Predict Tomorrow’s Stock Price
The script predicts the stock price for the next trading day using the RBF model, which is generally more suited for non-linear data patterns.
```
python

predicted_tomorrow_price = svr_rbf.predict([[tomorrow_index]])[0]
print(f"The predicted price for tomorrow is: {predicted_tomorrow_price}")
```
## How to Run the script
1. Clone or download this repository.
2. Install the required Python libraries using the command:
```
bash

pip install numpy matplotlib scikit-learn yfinance
```
3. Run the Python script:
```
bash

python svr_stock_prediction.py
```
### Expected Output
- The script will fetch Tesla stock data for the past two years, train the SVR models, and display a plot comparing the predicted prices against the actual stock prices.
- The script will print the predicted stock price for the next day using the RBF SVR model:
```
bash

The predicted price for tomorrow is: <predicted_price>
```
### Example Workflow:
- Train SVR models: The script automatically trains three SVR models (Linear, Polynomial, and RBF).
- Visualize the results: A plot of the actual vs predicted prices will be displayed.
- Predict the future price: The RBF model is used to predict the stock price for the next trading day.

## License

[MIT](https://choosealicense.com/licenses/mit/)

### Screenshot
![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)