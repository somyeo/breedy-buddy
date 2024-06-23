
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# 데이터 로드 함수
def load_data(file):
    data = pd.read_csv(file)
    data = data.loc[:, ~data.columns.str.startswith('Unnamed:')]
    if 'controller' in data.columns:
        data = data.drop(['controller'], axis=1)
    if 'date' in data.columns:
        data = data.drop(['date'], axis=1)
    return data

# 데이터 전처리 함수
def preprocess_data(data, target_col):
    X = data.drop(target_col, axis=1)
    y = data[target_col].values

    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y.astype(str).ravel())

    cat_columns = X.select_dtypes(include='object').columns
    num_columns = X.select_dtypes(exclude='object').columns

    X[cat_columns] = X[cat_columns].fillna("<NA>")
    medians = X[num_columns].median()
    X[num_columns] = X[num_columns].fillna(medians)

    encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    cat_values = encoder.fit_transform(X[cat_columns])
    num_values = preprocessing.StandardScaler().fit_transform(X[num_columns])

    X = np.hstack((cat_values, num_values))

    return X, y, label_encoder, list(cat_columns) + list(num_columns)

# 모델 훈련 함수
def train_model(X, y, model_type):
    if model_type == 'fan':
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=50, min_impurity_decrease=0.0)
    else:
        model = XGBClassifier(n_estimators=4, max_leaves=4 if model_type == 'watering' else 8, min_child_weight=1.0 if model_type == 'watering' else 0.037, learning_rate=0.1 if model_type == 'watering' else 0.067, subsample=1.0, colsample_bylevel=1.0 if model_type == 'watering' else 0.917, colsample_bytree=1.0, reg_alpha=0.001, reg_lambda=1.0 if model_type == 'watering' else 0.117)
    model.fit(X, y)
    return model

# Controller 컬럼 추가 함수
def add_controller_column(data, fan_model, watering_model, water_pump_model, fan_label_encoder, watering_label_encoder, water_pump_label_encoder, feature_names):
    for feature in feature_names:
        if feature not in data.columns:
            data[feature] = 0

    X = data[feature_names]
    fan_predictions = fan_label_encoder.inverse_transform(fan_model.predict(X))
    watering_predictions = watering_label_encoder.inverse_transform(watering_model.predict(X))
    water_pump_predictions = water_pump_label_encoder.inverse_transform(water_pump_model.predict(X))

    data['Fan_actuator'] = fan_predictions
    data['Watering_plant_pump'] = watering_predictions
    data['Water_pump_actuator'] = water_pump_predictions

    data['Controller'] = [
        f"{'Y' if fan == 'ON' else 'N'}_{'Y' if watering == 'ON' else 'N'}_{'Y' if water_pump == 'ON' else 'N'}"
        for fan, watering, water_pump in zip(fan_predictions, watering_predictions, water_pump_predictions)
    ]

    return data

# 웹 앱 인터페이스
st.title('IOT Controller Prediction')

st.sidebar.title('Upload your data')
train_file = st.sidebar.file_uploader('Upload train data', type='csv')
test_file = st.sidebar.file_uploader('Upload test data', type='csv')

if train_file and test_file:
    st.write('Train data and Test data uploaded successfully!')

    # 데이터 로드
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    st.write('Train Data Preview:')
    st.dataframe(train_data.head())

    st.write('Test Data Preview:')
    st.dataframe(test_data.head())

    # Fan_actuator 모델 학습 및 평가
    X_train, y_train, fan_label_encoder, feature_names = preprocess_data(train_data, 'Fan_actuator')
    fan_model = train_model(X_train, y_train, 'fan')

    # Watering_plant_pump 모델 학습 및 평가
    X_train, y_train, watering_label_encoder, _ = preprocess_data(train_data, 'Watering_plant_pump')
    watering_model = train_model(X_train, y_train, 'watering')

    # Water_pump_actuator 모델 학습 및 평가
    X_train, y_train, water_pump_label_encoder, _ = preprocess_data(train_data, 'Water_pump_actuator')
    water_pump_model = train_model(X_train, y_train, 'water_pump')

    # 테스트 데이터셋에 예측 및 컨트롤러 컬럼 추가
    test_data_with_predictions = add_controller_column(test_data, fan_model, watering_model, water_pump_model, fan_label_encoder, watering_label_encoder, water_pump_label_encoder, feature_names)

    st.write('Predicted Test Data with Controller:')
    st.dataframe(test_data_with_predictions)

    # CSV로 다운로드 링크 제공
    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(test_data_with_predictions)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='project_result.csv',
        mime='text/csv',
    )
