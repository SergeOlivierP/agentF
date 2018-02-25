import pandas as pd
import numpy as np


class Market:

    def __init__(self, file):
        self.rawData = pd.read_csv(file)
        self.asset_prices = self.rawData.IntelPriceUSD.values

        self.signals = self.rawData.iloc[:, 2:].values
        self.signals -= np.mean(self.signals, 0)
        self.signals /= np.std(self.signals, 0)

        self.duration = len(self.rawData.IntelPriceUSD)
