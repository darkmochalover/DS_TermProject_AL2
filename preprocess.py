import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
 
'''
Data Load
'''
df = pd.read_csv('data/combined_mbti_df.csv')
print(df[:][:5])
print(df.columns)

# Color Pallete (시각화에 필요함.)
mbti_color_pallete = dict({
    'INFP': '#48ad82',
    'ENFP': '#71bf9e',
    'INFJ': '#1a523a',
    'ENFJ': '#103123',
    
    'INTP': '#9874a6',
    'ENTP': '#af93ba',
    'INTJ': '#70507d',
    'ENTJ': '#46324e',
    
    'ISFP': '#e8b859',
    'ESFP': '#edc87e',
    'ISTP': '#b78d38',
    'ESTP': '#f5dfb5',
    
    'ISFJ': '#346f7b',
    'ESFJ': '#5ca9b8',
    'ISTJ': '#2c5f6a',
    'ESTJ': '#80bcc8',
    
})

fp_color_pallete = dict({
    'NF': '#34a474',
    'NT': '#8c649c',
    'SJ': '#4a9fb0',
    'SP': '#e5b046'
})

traits_color_palette = dict({
    'mind': {'Introvert': '#48ad82', 'Extrovert': '#8c649c', 'title': 'I vs E'},
    'energy_aspect': {'Observant': '#48ad82', 'Intuitive': '#8c649c', 'title': 'S vs N'},
    'nature': {'Feeling': '#48ad82', 'Thinking': '#8c649c', 'title': 'T vs F'},
    'tactics': {'Judging': '#48ad82', 'Prospecting': '#8c649c', 'title': 'J vs P'}
})
    
traits = [key for key in traits_color_palette]
mbti = [key for key in mbti_color_pallete]

'''
Preprocessing Part
'''
# MinMaxScaler Part
from sklearn.preprocessing import MinMaxScaler

audio_qualities = [
    'danceability_mean',
    'valence_mean',
    'energy_mean',
    'loudness_mean',
    'acousticness_mean',
    'instrumentalness_mean',
    'liveness_mean',
]

labels = [
    'mbti',
    'function_pair'
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

'''
Audio Quality를 MinMaxScaler를 통해 스케일링 함. 
(0부터 1까지의 범위)
'''
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

categories = renamed_columns[1:]
X.columns = renamed_columns


# 장조/단조의 개수의 합을 계산해서 저장 (C장조, D단조, .. 이렇게 따로 계산되는거 말고, 위에 지정된 list 이용해서 sum값 넣어줌)
X['major_count'] = df[major_tones].sum(axis=1)
X['minor_count'] = df[minor_tones].sum(axis=1)

print(X)

'''
mbti를 문자열이 아닌 LabelEncoder 통해 숫자로 바꿔준다.
'''
# 라벨 인코더 생성
encoder = LabelEncoder()
encoder.fit(df[['mbti']])
y = encoder.transform(df[['mbti']])
# print(y)
y_df = pd.DataFrame(y, columns=['mbti']) # 변환 완료
encoded_df = X.join(y_df)

print("[Encoded Dataframe]")
print(encoded_df)

## 확인용 프린트
print(encoded_df.columns) # Column 확인
print(encoded_df.sample(5)) # 샘플 출력

encoded_df.to_csv('data/encoded_data.csv')

# -----------------------------------------------------------------------
'''
Scaler Part
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=.1)

'''
StandardScaler
'''

std_scaler = StandardScaler()

std_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = std_scaler.transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = std_scaler.transform(X_test)

'''
MinMaxScaler
'''
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = minmax_scaler.transform(X_train)

# 테스트 데이터 스케일링
X_test_scaled = minmax_scaler.transform(X_test)

'''
RobustScaler 
'''

robust_scaler = RobustScaler()
robust_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = robust_scaler.transform(X_train)

# 테스트 데이터 스케일링
X_test_scaled = robust_scaler.transform(X_test)


'''
SMOTE Part
'''
# +) encoded_df랑 df의 라벨값은 인코딩 유무에만 차이가 있음, 시각화를 위해 출력은 df 이용
df['mbti'].value_counts().plot(kind='barh')
plt.show()

smote = SMOTE(sampling_strategy='auto', random_state=0)
X_train_over,y_train_over = smote.fit_resample(X_train,y_train)
print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)


# 라벨 디코딩 후 분포 확인
print('SMOTE 적용 후 레이블 값 분포')
decoded_labels = encoder.inverse_transform(y_train_over)
print(pd.Series(decoded_labels).value_counts())

