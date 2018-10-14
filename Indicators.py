import numpy as np
import pandas as pd

class Indicators:
    def __init__(self):


    def VolumeWeightedMA(sample_array, volume_array, N):
        return np.average(sample_array[-N:] * volume_array[-N:]) / np.average(volume_array[-N:])


    def SimpleMA(sample_array, N):
        data = pd.DataFrame(sample_array)
        return data.rolling(window=N).mean()
