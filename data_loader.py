import pandas as pd
import numpy as np
from datetime import datetime as dt

import warnings
warnings.simplefilter('ignore', FutureWarning)


class data_load():
    def __init__(self, company:str):       
        self.company = company
        self.data_path = 'market_data/'
        
    def load_from_csv(self, elements:list, start:str, end='2022/4/30'):
        # elements : columns
        # start, end: rows
        start_date = dt.strptime(start, '%Y/%m/%d')
        end_date = dt.strptime(end, '%Y/%m/%d')
        
        data_temp = pd.read_csv(self.data_path + str(self.company) + '_market_data.csv', header=0, encoding='Shift-JIS', usecols=lambda x: x in elements).iloc[::-1]
        data_temp.index = range(len(data_temp))
        data_temp['日付'] = pd.to_datetime(data_temp['日付'], format='%Y/%m/%d')
        data = data_temp[start_date <= data_temp['日付']]

        # self.data : pandas
        self.df = data[end_date >= data_temp['日付']]
        self.df.reset_index(inplace=True, drop=True)

if __name__== '__main__':
    print('It is a test. You can get the data of NTT.')

    data = data_load(company='NTT')        
    data.load_from_csv(elements=['日付', '始値', '高値', '安値', '終値'], start='2007/1/4')
    
    # print(data.df)
