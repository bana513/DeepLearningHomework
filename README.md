# Deep Learning Homework

Price prediction from OHLC data

**Team:** BotosBanda 

**Team members:** 

* Bial Bence - gbenceg@gmail.com
* Horváth István - histvan1995@gmail.com
* Máté Kristóf - mate.kry@gmail.com
 

**Project:** FIN1 

**Project description:**

We developed a deep neural network that can predict the price of a stock market element. This is a time series analysis task. According to the scholarly literature, the best way to accomplish the task is using *LSTM* components in the network. We consider Bitcoin to USDT exchange dataset as training dataset. We downloaded this dataset from *Poloniex*.


The features we use are the following: 
* Bitcoin candlestick OHLC values: open, high, low, close
* Volume
* BTC to ETH, BTC to XRP, BTC to LTC, BTC to XMR and BTC to DASH weighted averages


The dataset is since 2017-01-01T00:00:00+00:00 till 2018-10-14T20:00:00+00:00 because it contains enough information for the first sight and the trading volume is suitable.

## Learning

Run Learning.ipynb