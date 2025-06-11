import pandas as pd
import numpy as np

class OneHotEncoder:
    def __init__(self, data):
        self.data = data
        self.bin_unique_values = None
        self.unique_values = None

    def binary_one_hot_encode(self, column):
        self.bin_unique_values = column.unique()
        if len(self.bin_unique_values) != 2:
            raise ValueError("Column must have exactly two unique values for binary encoding.")
        return pd.concat([self.data.reset_index(drop=True), pd.DataFrame(columns=[f"one_hot_{column.name}"], data=np.where(column == self.bin_unique_values[0], 1, 0)).reset_index(drop=True)], axis=1).drop(column.name, axis=1)
    
    def one_hot_encode(self, column):
        self.unique_values = column.unique()
        if len(self.unique_values) <= 2:
            raise ValueError("Column must have at least three unique values, if you have two unique values, use `binary_one_hot_encode` instead.")
        tmp_df = pd.DataFrame(index=column.index, columns=self.unique_values)
        for value in self.unique_values:
            tmp_df[value] = np.where(column == value, 1, 0)
            # print(tmp_df, end='\n----------------------------------------------------------------------------------------------\n')
        return pd.concat([self.data, tmp_df], axis=1).drop(column.name, axis=1)