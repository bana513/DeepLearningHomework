import pandas as pd

df = pd.read_csv("poloniex_usdt_btc_20170101_DOHLCV_300.csv", sep=';')
sma = df.rolling(20).mean().values[:,4]
df = df.assign(sma=sma)
df. to_csv("poloniex_usdt_btc_20170101_DOHLCV_300_sma.csv", sep=";")

print(df)

