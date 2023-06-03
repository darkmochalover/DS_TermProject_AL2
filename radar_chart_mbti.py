import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import ipywidgets as widgets
from IPython.display import display
from utils import data_path

'''
Data Load
'''
df = pd.read_csv(data_path)
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
    'A#/Bbminor_count', 'A#/BbMajor_count', 'Bminor_count', 'BMajor_count'
]

df_agg = df.groupby(labels, as_index=False)[audio_qualities + all_tones].mean()

# Set the 'mbti' column as the index 
df_agg.set_index('mbti', inplace=True)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
df_agg[audio_qualities + all_tones] = scaler.fit_transform(df_agg[audio_qualities + all_tones])

'''
Radar Chart Part
'''
def draw_radar_chart(mbti):
    row = df_agg.loc[mbti, audio_qualities].values.flatten().tolist()
    row += row[:1]
    
    color = mbti_color_pallete[mbti]
    
    fig = plt.figure(figsize=(10, 10))
    
    angles = [n / float(len(audio_qualities)) * 2 * pi for n in range(len(audio_qualities))]
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], audio_qualities, color='grey', size=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1], ['0.2', '0.4', '0.6', '0.8', '1'], color="grey", size=20)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, row, linewidth=2, linestyle='solid', color=color)

    # Fill area
    ax.fill(angles, row, alpha=0.4, color=color)

    # Add MBTI label
    ax.text(0.5, 1.1, mbti, transform=ax.transAxes, ha='center', va='center', size=16)

    # Show the graph
    plt.show()

# Create a dropdown for selecting MBTI types
dd3 = widgets.Dropdown(options=mbti, description='MBTI')

# Create a dictionary of interactive outputs for each MBTI type
outputs = {}
for mbti_type in mbti:
    out = widgets.interactive_output(draw_radar_chart, {'mbti': widgets.fixed(mbti_type)})
    outputs[mbti_type] = out

# Create a grid of dropdowns and radar charts
grid = widgets.GridBox([dd3] + list(outputs.values()), layout=widgets.Layout(grid_template_columns="repeat(4, auto)"))
display(grid)

def update_dropdown(change):
    # Update the selected MBTI type in all dropdowns
    selected_mbti = change.new
    for dropdown in outputs:
        dropdown.value = selected_mbti

# Link the value of the main dropdown to all other dropdowns
dd3.observe(update_dropdown, 'value')
