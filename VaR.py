# Using Monte Carlo simulation
# Using the variance-covariance method

import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import datetime as dt

# Create our portfolio of equities
tickers = ['AAPL','META', 'C', 'DIS']

# Set the investment weights (I arbitrarily picked for example)
weights = np.array([.25, .3, .15, .3])

# Set an initial investment level
initial_investment = 1000000

# Download closing prices
data = ydata = yf.download(tickers, start="2018-01-01", end=dt.date.today())['Close']

#From the closing prices, calculate periodic returns
returns = data.pct_change()
returns = returns.dropna()

returns.tail()

# Generate Var-Cov matrix
cov_matrix = returns.cov()
cov_matrix


# Calculate mean returns for each stock
avg_rets = returns.mean()

# Calculate mean returns for portfolio overall, 
# using dot product to 
# normalize individual means against investment weights
 # https://en.wikipedia.org/wiki/Dot_product#:~:targetText=In%20mathematics%2C%20the%20dot%20product,and%20returns%20a%20single%20number.
port_mean = avg_rets.dot(weights)

# Calculate portfolio standard deviation
port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))

# Calculate mean of investment
mean_investment = (1+port_mean) * initial_investment

# Calculate standard deviation of investmnet
stdev_investment = initial_investment * port_stdev


# Select our confidence interval (I'll choose 95% here)
conf_level1 = 0.05

# Using SciPy ppf method to generate values for the
# inverse cumulative distribution function to a normal distribution
# Plugging in the mean, standard deviation of our portfolio
# as calculated above
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

from scipy.stats import norm, normaltest, invgauss
cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)

#Finally, we can calculate the VaR at our confidence interval
var_1d1 = initial_investment - cutoff1
var_1d1
normaltest(returns['AAPL'])
# p <= alpha: reject H0, not normal.
# p > alpha: fail to reject H0, normal.

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# Repeat for each equity in portfolio
returns['META'].hist(bins=40, density=True,histtype="stepfilled",alpha=0.5)
x = np.linspace(port_mean - 3*port_stdev, port_mean+3*port_stdev,100)
plt.plot(x, norm.pdf(x, port_mean, port_stdev), "r")
plt.title("AAPL returns (binned) vs. normal distribution")
plt.show()

#EWMA

l = .97
EWMA = pd.DataFrame()
EWMA['log_ret'] = np.log(data['AAPL']) - np.log(data['AAPL'].shift(1))
EWMA['weight'] = np.arange(0,len(EWMA))
EWMA = EWMA.dropna()

EWMA_Variance =EWMA['log_ret'].ewm(com=.97,adjust=False).mean()
EWMA_Variance.plot()
