import numpy as np

class SpectroscopyData:
    def __init__(self, eis_data:np.ndarray):
        '''
        :param eis_data : raw data output from the PHOBOS acquisition system
        '''

        #check if the raw electrode data is a numpy array
        if type(eis_data) != np.ndarray:
            raise TypeError(f'[SpectroscopyData] Raw electrode data must be a numpy array! Curr. type = {type(eis_data)}')

        self.freq = eis_data[:,0]
        # self.Z_complex = eis_data[:,4]
        self.Z_real = eis_data[:,5]
        self.Z_imag = -eis_data[:,6]
