# Deep Learning Homework
Price prediction from OHLC data

**Team:** BotosBanda 

**Team members:** 

* Bial Bence - gbenceg@gmail.com
* Horváth István - histvan1995@gmail.com
* Máté Kristóf - mate.kry@gmail.com
 

**Project:** FIN1 

**Project description:**

We are planning a neural network that can predict the price of a stock market element. This is a time series analysis task. According to the scholarly literature, the best way to accomplish the task is using *LSTM* components in the network. We consider Bitcoin to USDT exchange dataset as training dataset. We downloaded this dataset from *Poloniex*.


The features we use are the following: 
* Raw OHLC values
* Different averaged OHLC values in 5 minutes long interval 
* Volume 


The dataset is since 2017-01-01T00:00:00+00:00 till 2018-10-14T20:00:00+00:00 because it contains enough information for the first sight and the trading volume is suitable.

## Learning and Prediction
Preprocess.ipynb is not necessary for this part yet.
First run Learning.ipynb (Simplified version of Preprocess.ipynb because the input data was too complex to learn something userful with simple NN).
Now you can tun Prediction.ipynb.