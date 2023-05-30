import pandas as pd
import numpy as np

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
centers_df = df.iloc[: , :22] # 처음 22개 열을 선택해서 저장
centers_df = df[audio_qualities] # 오디오 품질과 해당 열이 있는 열을 선택해서 저장

centers_df_scaled = MinMaxScaler().fit_transform(centers_df) # MinMaxScaler 사용
df_scaled = pd.DataFrame(centers_df_scaled, columns=[audio_qualities])

labels_df = df[labels]
aq_df = labels_df.join(df_scaled) # 스케일링한 데이터를 aq_df에 join함

# 열 이름을 바꾸기 위해 리스트로 저장함
renamed_columns =  [
    'mbti',
    'function_pair',
    'danceability',
    'valence',
    'energy',
    'loudness',
    'acousticness',
    'instrumentalness',
    'liveness'
]

categories = renamed_columns[2:]
aq_df.columns = renamed_columns


# 장조/단조의 개수의 합을 계산해서 저장 (C장조, D단조, .. 이렇게 따로 계산되는거 말고, 위에 지정된 list 이용해서 sum값 넣어줌)
aq_df['major_count'] = df[major_tones].sum(axis=1)
aq_df['minor_count'] = df[minor_tones].sum(axis=1)

# 각 MBTI에 맞는 설명 label 붙이는 부분
aq_df['mind'] = np.where(aq_df['mbti'].str.contains('I'), 'Introvert', 'Extrovert')
aq_df['energy_aspect'] = np.where(aq_df['mbti'].str.contains('N'), 'Intuitive', 'Observant')
aq_df['nature'] = np.where(aq_df['mbti'].str.contains('F'), 'Feeling', 'Thinking')
aq_df['tactics'] = np.where(aq_df['mbti'].str.contains('P'), 'Prospecting', 'Judging')

## 확인용 프린트
print(aq_df.columns) # Column 확인
print(aq_df.sample(5)) # 샘플 출력


'''
Preprocessing하여 얻은 데이터프레임을 csv파일로 다시 저장함.
'''
aq_df.to_csv('data/preprocessed_mbti_data.csv') # 데이터프레임을 csv 저장