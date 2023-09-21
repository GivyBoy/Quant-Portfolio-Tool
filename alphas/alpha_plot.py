import alpha_functions as alphas

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

data = yf.download('^GSPC', start='2010-01-01')

alpha_data = alphas.alpha_009(data)
print(alpha_data)
# Plot the alpha values
plt.plot(alpha_data)
plt.xlabel('Date')
plt.ylabel('Alpha 9')
plt.title('Alpha 9')
plt.show()