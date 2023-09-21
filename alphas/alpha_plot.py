import alpha_functions as alphas

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download('^GSPC', start='2010-01-01')

alpha_006_values = alphas.alpha_006(data)

# Plotting the alpha
plt.figure(figsize=(10,6))
plt.plot(alpha_006_values)
plt.title('Alpha 6 Output')
plt.tight_layout()
plt.show()


