import os
import math
import random
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pvlib.location import Location

# 設定隨機種子以確保結果可重現
SEED = 5397
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def get_geodistance(lon1, lat1, lon2, lat2):
    """計算兩點間的地理距離"""
    radians = map(math.radians, [lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = radians
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    distance = 2 * math.asin(math.sqrt(a)) * 6371 * 1000  # 地球半徑 * 1000 轉換成公尺
    return distance

def preprocess_data(df):
    """數據預處理，包括時間轉換和單位換算"""
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Irradiance'] /= 3.6  # 1 kwh = 3.6 MJ
    df['CapacityFactor'] = df['Generation'] / df['Capacity']
    df['ArrayRatio'] = df['CapacityFactor'] / df['Irradiance']
    return df

def group_data(df, by):
    """按照特定條件進行分組並計算平均值"""
    grouped = df.groupby(by).mean().reset_index().sort_values(by=by, ascending=False)
    return grouped

def model_training_and_prediction(train, test, features, target):
    """模型訓練和預測"""
    X_train, X_test, y_train, y_test = train_test_split(train[features], train[target], test_size=0.2, random_state=SEED)
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=SEED)
    model.fit(X_train, y_train)
    predictions = model.predict(test[features])
    return predictions

def main():
    """主函數，執行數據加載、預處理、模型訓練和預測"""
    path = "."
    train = pd.read_csv(os.path.join(path, 'data/train.csv'))
    test = pd.read_csv(os.path.join(path, 'data/test.csv'))
    
    # 數據預處理
    train = preprocess_data(train)
    test = preprocess_data(test)

    # 特徵和目標列
    features = ['Lat', 'Lon', 'Irradiance', 'Capacity']
    target = 'CapacityFactor'
    
    # 模型訓練和預測
    test['CapacityFactor_pred'] = model_training_and_prediction(train, test, features, target)
    
    # 結果視覺化
    plt.figure(figsize=(10, 6))
    plt.scatter(test['Lat'], test['CapacityFactor_pred'], c='blue', label='Predicted Capacity Factor')
    plt.xlabel('Latitude')
    plt.ylabel('Capacity Factor')
    plt.title('Predicted Capacity Factor by Latitude')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
