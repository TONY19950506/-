import os
import math
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 初始化隨機種子
SEED = 5397
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def preprocess_data(file_path):
    """數據預處理函數"""
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Irradiance'] /= 3.6
    data['Irradiance_m'] /= 1000
    data['CapacityFactor'] = data['Generation'] / data['Capacity']
    data['ArrayRatio'] = data['CapacityFactor'] / data['Irradiance']
    data['ArrayRatio_m'] = data['CapacityFactor'] / data['Irradiance_m']
    return data

def group_data(data, group_keys, agg_methods):
    """根據指定鍵和聚合方法對數據進行分組"""
    grouped = data.groupby(group_keys).agg(agg_methods).reset_index()
    return grouped.sort_values(by=group_keys, ascending=False)

def visualize_data(data, groupby_keys, ylabel='Normalized [0, 1]', figsize=(28, 28), suptitle=''):
    """數據視覺化"""
    grouped = group_data(data, groupby_keys, {'Generation': 'size', 'ArrayRatio': 'mean', 'CapacityFactor': 'mean'})
    plt.figure(figsize=figsize)
    plt.suptitle(suptitle, fontsize=24, y=0.91)

    for i, row in grouped.iterrows():
        plt.subplot(math.ceil(len(grouped) / 2), 2, i + 1)
        title = f"{row[groupby_keys[0]]}, {row[groupby_keys[1]]} [{row['Generation']}]"
        plt.title(title)
        plt.plot(data['Date'], data['Irradiance'] / data['Irradiance'].max(), label='Irradiance')
        plt.plot(data['Date'], data['Generation'] / data['Generation'].max(), label='Generation')
        plt.ylabel(ylabel)
        plt.legend()

# 主程序
if __name__ == "__main__":
    path = "."
    train_file = os.path.join(path, 'data/train.csv')
    test_file = os.path.join(path, 'data/test.csv')

    train_data = preprocess_data(train_file)
    test_data = preprocess_data(test_file)

    # 示範如何使用分組和視覺化函數
    group_keys = ['Lat', 'Lon']
    agg_methods = {'Generation': 'size', 'ArrayRatio': 'mean', 'CapacityFactor': 'mean'}
    train_grouped = group_data(train_data, group_keys, agg_methods)
    visualize_data(train_data, group_keys, suptitle='Irradiance and Generation of each Location')
