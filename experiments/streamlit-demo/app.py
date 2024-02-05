# note that if you want to change the name of this file you should also change the name in the `Procfile`
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

st.write("simple stock price app")

ticker = st.text_input("Enter a stock ticker")

if ticker or (st.button("Get Stock Price") and ticker != ""):
    data = yf.download(ticker, start="2010-01-01")["Adj Close"]

    fig, ax = plt.subplots()
    ax.plot(data)
    st.pyplot(fig)
