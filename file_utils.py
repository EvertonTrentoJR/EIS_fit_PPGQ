from utils import data_types
import pandas as pd
import csv
import numpy as np
import os

def read(filename:str):
    '''
    :param filename: path where the .csv is stored
    '''

    #check if the filename exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'[file_utils] Filename {filename} does not exist!')

    #process the raw data output  into a custom data structure
    raw_data = pd.read_csv(
        filename,
        delimiter=",",
        skiprows=5,
        encoding="utf-16"
    ).to_numpy()

    raw_data = raw_data[:, :-1]

    data = data_types.SpectroscopyData(raw_data)

    return data