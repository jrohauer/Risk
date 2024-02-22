import pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import scipy.optimize as sco


import yfinance as yf
from yahoofinancials import YahooFinancials
tickers = ['AEPGX','ANWPX','ABALX','AAANX','EEIAX','VYGQX','EVIBX','SDHY','NOBL']
weights = np.array([.08,.04,.01,.14,.21,.11,.11,.15,.15])

rets_df = pandas.DataFrame()
for i in tickers:
    temp = yf.download(i, start='2019-03-04', end='2022-03-04', progress=False)
    rets_df[i]=temp['Close']

   
rets_df = rets_df.pct_change(periods=1)
rets_df=rets_df.dropna()
cov = rets_df 


rets_df = rets_df+1
rets=(rets_df.product()-1)
rets=(1+rets)**(1/5)-1
rets=rets

M = rets

ones = np.ones(len(tickers))
sigma = np.cov(cov,rowvar=False)
sigma_inv = np.linalg.inv(sigma)


a =  np.matmul(np.matmul(ones.T,sigma_inv),ones)
b =  np.matmul(np.matmul(M,sigma_inv),ones)
c =  np.matmul(np.matmul(M.T,sigma_inv),M)

#Create Frontier

eff_frontier=pandas.DataFrame()
for m in np.linspace(-.1,.25,1000):
    l1 = (c-b*m)/(a*c-b**2)
    l2 = (a*m-b)/(a*c-b**2)
    pi = l1*np.matmul(sigma_inv,ones)+l2*np.matmul(sigma_inv,M)
    
    port_ret=np.matmul(pi,rets)
    port_vol = np.matmul(np.matmul(pi.T,sigma),pi)**.5*np.sqrt(252)
    temp_df=pandas.DataFrame(index=[m],data={'return':port_ret,'Volatility':port_vol})
    eff_frontier=eff_frontier.append(temp_df)
    
import seaborn as sns
sns.set_theme()    

x=eff_frontier['Volatility']
y=eff_frontier['return']
plt.plot(x,y)


x1=np.matmul(np.matmul(weights.T,sigma),weights)**.5*np.sqrt(252)
y1 = np.matmul(weights,rets)
plt.plot(x1,y1, marker=".", markersize=20)
