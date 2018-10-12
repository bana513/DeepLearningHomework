import numpy as np

class Indicators:
    def __init__(self):


    def VolumeWeightedMA(sample_array, volume_array, N):
        return np.average(sample_array[-N:] * volume_array[-N:]) / np.average(volume_array[-N:])




