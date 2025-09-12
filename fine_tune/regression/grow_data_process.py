import os
import pandas as pd
import numpy as np



if __name__ == '__main__':
    os.chdir('../data')
    import pandas as pd
    with open("processed_data.txt", 'r') as file:
        new_column_order = file.readline().strip().split('\t')
    data_df = pd.read_csv("yeast9_pnas_data.csv")
    # data_df['log2relT'] = np.log2(data_df['log2relT'] + 1)
    # data_df['log2relT'] = 2 ** data_df['log2relT']
    final_df = pd.DataFrame({col: data_df[col] if col in data_df.columns else 0 for col in new_column_order})
    # min_val = final_df.min().min()
    # max_val = final_df.max().max()
    # max_val_05_15 = final_df[final_df <= 1.5].max().max()
    # min_val_15 = final_df[final_df > 1.5].min().min()
    # max_val_05 = final_df[final_df <= 0.5].max().max()
    # min_val_05_15 = final_df[final_df > 0.5].min().min()
    # final_df = final_df.applymap(lambda x: (x - min_val_15) / (max_val - min_val_15) * (799 - 700) + 700 if x > 1.5 else x)
    # final_df = final_df.applymap(lambda x: (x - min_val_05_15) / (max_val_05_15 - min_val_05_15) * (699-100) + 100 if 0.5 < x <= 1.5 else x)
    # final_df = final_df.applymap(lambda x: (x - min_val) / (max_val_05 - min_val) * 99 if x <= 0.5 else x)
    # final_df['log2relT'] = [
    #     [1, 0, 0] if x < 0 else [0, 1, 0] if x == 0 else [0, 0, 1]
    #     for x in data_df['log2relT']
    # ]
    final_df['log2relT'] = data_df['log2relT']
    final_df.to_csv("processed_yeast9_pnas_data_ori.csv", index=False)


