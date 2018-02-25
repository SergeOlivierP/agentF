import pandas as pd
import numpy as np


class Market:

    def __init__(self, file, number_of_assets=1):
        self.rawData = pd.read_csv(file)
        self.number_of_assets = number_of_assets
        # Compute number of signals. Remove 1 to account for the date column (first column)
        self.number_of_signals = len(self.rawData.values[0,:]) - number_of_assets - 1 
        self.duration = len(self.rawData.values[:,1])

        self.signals = self.rawData.iloc[:, 2:].values
        self.signals = self.preprocess_data(self.signals)

        # temporarily force coded the cash asset
        # self.asset_prices = self.rawData.IntelPriceUSD.values
        self.asset_prices = np.append(np.ones([self.duration,1]), self.rawData.values[:,1:number_of_assets+1].reshape(self.duration,number_of_assets))

    def preprocess_data(signals) # input := np.array
        # thats where we should modify the signals, e.g. standardization
        signals -= np.mean(signals, 0)
        signals /= np.std(signals, 0)
        pass