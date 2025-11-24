import os
import pandas as pd
import numpy as np


if __name__ == '__main__':
    os.chdir('../data')
    # 0 - 等渗，1 - 低渗，2 - 氨基酸，3 - 葡萄糖
    original_data = pd.read_csv('pressure_data.csv')
    scale_data = np.log(original_data.select_dtypes(include=[np.number]) + 1)
    scale_data = scale_data.select_dtypes(include=[np.number])/2.
    # scale_data = (scale_data * 100).round().astype(int)
    scale_data['Gene'] = [name.replace('-', '.') for name in original_data['Geneid']]
    cols = ['Gene'] + [col for col in scale_data.columns if col != 'Gene']
    scale_data = scale_data[cols]
    scale_data.to_csv('log(tpm+1)d2_pressure_data_1.csv', index=False)
    print(1)