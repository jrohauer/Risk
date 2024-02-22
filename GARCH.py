import numpy as np
from arch import arch_model
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt



tickers = ['AAPL']

data = yf.download(tickers, start="2018-01-01", end=dt.date.today())['Close']
returns = pd.DataFrame(np.diff(np.log(data.values)))
returns.index = data.index.values[1:data.index.values.shape[0]]
returns.columns = ['AAPL Returns']

plt.figure(figsize=(15,5));
plt.plot(returns.index,returns);
plt.ylabel('Returns');
plt.title('AAPL Returns');


am = arch_model(returns)


from arch import ConstantMean, GARCH, Normal

am = ConstantMean(returns)
am.volatility = GARCH(1, 0, 1)
am.distribution = Normal()