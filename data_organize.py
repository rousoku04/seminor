import numpy as np
from data_loader import data_load
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.simplefilter('ignore', FutureWarning)


def normalization(vec):
    vec_max = vec.max()
    return vec / vec_max, vec_max


class data_org():
    def __init__(self, company: str, elements: list, start: str):
        self.dt = data_load(company=company)
        self.dt_test = data_load(company=company)
        
        self.elements = elements

        # train
        self.dt.load_from_csv(elements=self.elements, start=start)

        # test : start = 2022/09/01, end = 2022/11/01
        self.dt_test.load_from_csv(elements=self.elements, start='2022/5/1', end='2022/11/1')


        for _, element in enumerate(self.elements):
            if element == '日付':
                self.num_days = len(self.dt.df[element])
                self.input = np.arange(self.num_days)
                self.num_days_test = len(self.dt_test.df[element])
                self.input_test = np.arange(self.num_days_test) + self.num_days 

                continue

            # str → remove comma → float
            self.dt.df[element] = self.dt.df[element].str.replace(',', '').astype(float).values
            self.dt_test.df[element] = self.dt_test.df[element].str.replace(',', '').astype(float).values


    def time_variable(self):
        elements = self.elements

        if '日付' in elements:
            elements.remove('日付')

        # Train
        input = []
        output = []

        for i in range(self.num_days):
            input_temp = []
            input_temp.append(i)
            for _, element in enumerate(elements):
                if element == '始値':
                    continue
                else:
                    input_temp.append(self.dt.df.loc[i, element])

            output_temp = self.dt.df.loc[i, ['始値']].values


            input.append(input_temp)
            output.append(output_temp)

            if i % int(self.num_days / 10) == 0:
                print('.')

        self.input = np.array(input)
        self.output = np.array(output)

        self.output_nor, self.output_max = normalization(self.output)

        # Test
        input_test = []
        output_test = []

        for j in range(self.num_days_test):
            input_temp = []
            input_temp.append(self.input_test[j])
            for _, element in enumerate(elements):
                if element == '始値':
                    continue
                
                else:
                    input_temp.append(self.dt_test.df.loc[j, element])

            output_temp = self.dt_test.df.loc[j, ['始値']].values

            input_test.append(input_temp)
            output_test.append(output_temp)

            if j %  int(self.num_days_test / 10) == 0:
                print('.')

        self.input_test = np.array(input_test)
        self.output_test = np.array(output_test)

        self.output_test_nor = self.output_test / self.output_max

    def time_sequence(self, delay:int):
        elements = self.elements
        if '日付' in elements:
            elements.remove('日付')

        if delay > self.num_days:
            print('Delay is too big...')

        input = []
        output = []

        for i in range(self.num_days - delay):

            input_temp = self.dt.df.loc[i:i+delay-1, elements].values
            output_temp = self.dt.df.loc[i+delay, ['始値']].values

            input.append(input_temp)
            output.append(output_temp)

            if i % int((self.num_days - delay) / 10) == 0:
                print('.')

        self.input_seq = np.array(input)
        self.output_seq = np.array(output)

        self.output_seq_nor, self.output_max = normalization(self.output_seq)
        self.input_seq_nor = self.input_seq / self.output_max

        self.output_test = np.array(self.dt_test.df.loc[:, elements].values)
        self.output_test_nor = self.output_test / self.output_max

if __name__ == '__main__':

    data = data_org(company='NTT', elements=['日付', '始値', '高値', '安値', '終値'], start='2007/1/4')
    data.time_variable()
    data.time_sequence(delay=4)