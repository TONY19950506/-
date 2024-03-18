import os
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

# 初始化隨機種子
SEED = 5397
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def preprocess_data(filepath):
    """預處理數據"""
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Irradiance'] /= 3.6  # MJ to kWh
    data['Irradiance_m'] /= 1000  # W/m^2 to kW/m^2
    data['CapacityFactor'] = data['Generation'] / data['Capacity']
    data['ArrayRatio'] = data['CapacityFactor'] / data['Irradiance']
    data['ArrayRatio_m'] = data['CapacityFactor'] / data['Irradiance_m']
    return data

def aggregate_data(df, by, agg_method='mean'):
    """根據指定列分組聚合"""
    return df.groupby(by).agg(agg_method).reset_index().sort_values(by=by, ascending=False)

def calculate_generation_based_on_array_ratio(df, agg_df):
    """根據ArrayRatio計算Generation"""
    df = df.merge(agg_df[['Lat', 'Lon', 'Module', 'Capacity', 'ArrayRatio']], on=['Lat', 'Lon', 'Module', 'Capacity'], how='left')
    df['CalculatedGeneration'] = df['Irradiance'] * df['Capacity'] * df['ArrayRatio']
    return df

def remove_outliers(df, outliers):
    """移除指定的異常值"""
    for i, outlier_dates in enumerate(outliers):
        df = df[~df['Date'].isin(outlier_dates['irradiance'])]
        df = df[~df['Date'].isin(outlier_dates['generation'])]
    return df

# 主程式
if __name__ == "__main__":
    path = "."
    train_data = preprocess_data(os.path.join(path, 'data/train.csv'))
    test_data = preprocess_data(os.path.join(path, 'data/test.csv'))
    
    # 移除訓練數據中的異常值
    outliers = [
        {'irradiance': [], 'generation': ['2021-09-10']},
        # 添加其他異常值...
    ]
    train_data_cleaned = remove_outliers(train_data, outliers)
    
    # 計算分組平均ArrayRatio
    agg_train_data = aggregate_data(train_data_cleaned, ['Lat', 'Lon', 'Module', 'Capacity'])
    
    # 使用平均ArrayRatio計算測試數據的生成量
    test_data_with_generation = calculate_generation_based_on_array_ratio(test_data, agg_train_data)
    
    # 保存結果
    test_data_with_generation[['ID', 'CalculatedGeneration']].to_csv(os.path.join(path, 'submission/generation.csv'), index=False)
    
    # 畫出一些結果（視需要調整或刪除此部分）
    plt.figure(figsize=(10, 5))
    plt.plot(test_data_with_generation['Date'], test_data_with_generation['CalculatedGeneration'], label='Calculated Generation')
    plt.xlabel('Date')
    plt.ylabel('Generation')
    plt.title('Test Data Generation Prediction')
    plt.legend()
    plt.show()
