import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from Historic_Crypto import HistoricalData

class CryptoData:
    def __init__(self):
        pass

    def get_crypto_data(self, start_date, end_date, split_date):
        try:
            train = pd.read_csv("Datasets/train_eth.csv", index_col=0)
            test = pd.read_csv("Datasets/test_eth.csv", index_col=0)
        except:
            df = HistoricalData('ETH-USD', 3600, start_date, end_date).retrieve_data()
            split_row = df.index.get_loc(split_date)
            cols = df.columns.tolist()
            df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=cols)
            train = df[:split_row]
            test = df[split_row:]
            train.to_csv("Datasets/train_eth.csv")
            test.to_csv("Datasets/test_eth.csv")
        return train, test

    def get_precollected_data(self, split_ratio=0.8):
        historic_data = pd.read_csv("Datasets/eth_combined_hourly_daily.csv", index_col=0)
        if historic_data.index.name != 'Date':
            historic_data['Date'] = pd.to_datetime(historic_data['Date'], infer_datetime_format=True)
            historic_data = historic_data.set_index('Date')
        historic_data = historic_data.dropna()
        cols = historic_data.columns.tolist()
        train_unscaled, test_unscaled = train_test_split(historic_data, train_size=split_ratio,shuffle=False)
        train_scaled = pd.DataFrame(MinMaxScaler().fit_transform(train_unscaled), columns=cols)
        test_scaled = pd.DataFrame(MinMaxScaler().fit_transform(test_unscaled), columns=cols)
        train_unscaled = pd.DataFrame(train_unscaled,  columns=cols)
        test_unscaled = pd.DataFrame(test_unscaled,  columns=cols)
        train_unscaled = train_unscaled[['Low', 'High', 'Open', 'Close', 'Yesterday_Open', 'Yesterday_High', 'Yesterday_Low', 'Yesterday_Close']]
        test_unscaled = test_unscaled[['Low', 'High', 'Open', 'Close', 'Yesterday_Open', 'Yesterday_High', 'Yesterday_Low', 'Yesterday_Close']]
        train_unscaled.to_csv("Output/train_unscaled.csv")
        train_scaled.to_csv("Output/train_scaled.csv")
        test_unscaled.to_csv("Output/test_unscaled.csv")
        test_scaled.to_csv("Output/test_scaled.csv")
        return train_scaled, test_scaled, train_unscaled, test_unscaled
