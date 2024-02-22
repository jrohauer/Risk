import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

a = 4
mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')

x = np.linspace(skewnorm.ppf(0.01, a),
                skewnorm.ppf(0.99, a), 100)

#skewnorm takes a real number  as a skewness parameter
ax.plot(x, skewnorm.pdf(x, a),'r-', lw=5, alpha=0.6, label='skewnorm pdf')



#loc = mean
#scale = stdev
#skew = skew
import scipy.stats as ss
import numpy as np
#Note scale is standard deviation in scipy stats
sample = ss.pearson3.rvs(loc=50, scale=2, skew=-1,  size=1000)
print(np.mean(sample),
np.var(sample),
ss.skew(sample),
ss.kurtosis(sample))


import statsmodels.sandbox.distributions.extras as extras
import scipy.interpolate as interpolate
import scipy.stats as ss
import matplotlib.pyplot as plt  
import numpy as np

def generate_normal_four_moments(mu, sigma, skew, kurt, size=10000, sd_wide=10):
   f = extras.pdf_mvsk([mu, sigma, skew, kurt])
   x = np.linspace(mu - sd_wide * sigma, mu + sd_wide * sigma, num=500)
   y = [f(i) for i in x]
   yy = np.cumsum(y) / np.sum(y)
   inv_cdf = interpolate.interp1d(yy, x, fill_value="extrapolate")
   rr = np.random.rand(size)

   return inv_cdf(rr)

data = generate_normal_four_moments(mu=0, sigma=1, skew=-1, kurt=3)