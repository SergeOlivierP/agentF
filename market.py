import pandas as pd
import numpy as np


class Market:

    def __init__(self, file):
        self.rawData = pd.read_csv(file)
        self.stock_price = self.rawData.IntelPriceUSD.values

        self.indices = self.rawData.iloc[:, 2:].values
        self.indices -= np.mean(self.indices, 0)
        self.indices /= np.std(self.indices, 0)

        self.duration = len(self.rawData.IntelPriceUSD)
