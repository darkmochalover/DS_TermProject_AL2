import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from utils import raw_data_path, data_path
import itertools


'''
Data Load
'''
df = pd.read_csv(raw_data_path)
# print(df[:][:5])
# print(df.columns)


'''
Dictionary
'''

##############
# Dictionary #
##############


audio_qualities = [
    'danceability_mean',
    'valence_mean',
    'energy_mean',
    'loudness_mean',
    'acousticness_mean',
    'instrumentalness_mean',
    'liveness_mean',
]

# 장조/단조 (Major/Minor)
all_tones = [
    'Cminor_count', 'CMajor_count', 'C#/Dbminor_count', 'C#/DbMajor_count',
    'DMajor_count', 'D#_EbMajor_count', 'Eminor_count', 'EMajor_count',
    'Fminor_count', 'FMajor_count', 'F#/Gbminor_count', 'GMajor_count',
    'G#/Abminor_count', 'G#/AbMajor_count', 'Aminor_count', 'AMajor_count',
    'A#/Bbminor_count', 'BMajor_count', 'Dminor_count', 'D#_Ebminor_count',
    'Gminor_count', 'A#/BbMajor_count', 'F#/GbMajor_count', 'Bminor_count'
]

major_tones = [
    'CMajor_count', 'C#/DbMajor_count',
    'DMajor_count', 'D#_EbMajor_count', 
    'EMajor_count',
    'FMajor_count', 
    'GMajor_count', 'G#/AbMajor_count', 
    'AMajor_count', 'BMajor_count', 'A#/BbMajor_count', 
    'F#/GbMajor_count'
]
minor_tones = [
    'Cminor_count', 'C#/Dbminor_count', 
    'Eminor_count', 
    'Fminor_count', 'F#/Gbminor_count', 
    'G#/Abminor_count',  
    'Aminor_count', 'A#/Bbminor_count', 
    'Dminor_count', 'D#_Ebminor_count',
    'Gminor_count', 
    'Bminor_count'
]




#######################
# Feature Engineering #
#######################

# Subset only measures of centers
X = df.iloc[: , :22] # 처음 22개 열을 선택해서 저장
X = df[audio_qualities] # 오디오 품질과 해당 열이 있는 열을 선택해서 저장

# 열 이름을 바꾸기 위해 리스트로 저장함
renamed_columns =  [
    'danceability',
    'valence',
    'energy',
    'loudness',
    'acousticness',
    'instrumentalness',
    'liveness'
]

categories = renamed_columns[:]
X.columns = renamed_columns


# 장조/단조의 개수의 합을 계산해서 저장 (C장조, D단조, .. 이렇게 따로 계산되는거 말고, 위에 지정된 list 이용해서 sum값 넣어줌)
X['major_count'] = df[major_tones].sum(axis=1).astype('int64')
X['minor_count'] = df[minor_tones].sum(axis=1).astype('int64')


def Encoding(df, encoding_method):
    df = df.copy()

    if(encoding_method == 'LabelEncoder'):
        encoder = LabelEncoder()
        target = encoder.fit_transform(df[['mbti']])
        

    if(encoding_method == 'OneHotEncoder'):
        encoder = OneHotEncoder(sparse=False)
        target = encoder.fit_transform(df[['mbti']])

    return target

def Scaling(scale_method, X_train, X_test):
    if( scale_method == 'No Scale'):
        return X_train, X_test

    elif(scale_method == 'StandardScaler'):
        scaler = StandardScaler()

    elif(scale_method == 'MinMaxScaler'):
        scaler = MinMaxScaler()

    elif(scale_method == 'RobustScaler'):
        scaler = RobustScaler()


    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    return X_train_scaled, X_test_scaled



encoder_list = ['LabelEncoder', 'OneHotEncoder']
scaler_list = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'No Scale']
test_size_list = [0.3, 0.2, 0.1]
do_smote = [True, False]

combinations = list(itertools.product(encoder_list, scaler_list, test_size_list, do_smote))

for encoder, scaler, test_size, do_smote in combinations:
    print(f"Encoder: {encoder}, Scaler: {scaler}, Test Size: {test_size}, Smote Phase: {do_smote}")

    y = Encoding(df = df, encoding_method=encoder)
    # print(y[:5])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5, test_size = test_size)

    X_train, X_test = Scaling(scale_method=scaler, X_train=X_train, X_test=X_test)

    if(do_smote): # If true,
        smote = SMOTE(sampling_strategy='auto', random_state=0)
        X_train, y_train = smote.fit_resample(X_train,y_train)

    print('Train 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)



    
