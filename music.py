import pandas as pd
import numpy as np

song_data = pd.read_csv('H:/Music_recommendation_knn/music_info/song_data.csv')
song_info = pd.read_csv('H:/Music_recommendation_knn/music_info/song_info.csv')
song_info[song_info['song_name'] == 'Footloose']
song_info['playlist'] = song_info['playlist'].str.lower()
song_info['song_name'] = song_info['song_name'].str.lower()
song_info['artist_name'] = song_info['artist_name'].str.lower()
song_info['album_names'] = song_info['album_names'].str.lower()

song_data['song_name'] = song_data['song_name'].str.lower()
song_df = song_data.copy()
song_df['artist_name'] = song_info['artist_name']

song_df = song_df.drop_duplicates(subset=['song_name', 'artist_name'])

song_info = song_info.drop_duplicates(subset=['song_name', 'artist_name'])

song_df = song_df.reset_index()
song_df = song_df.drop('index', axis=1)

song_df.head()

from scipy.stats import shapiro

stat, p = shapiro(song_df[['song_popularity', 'acousticness',
                           'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                           'loudness', 'audio_mode', 'speechiness', 'tempo', 'audio_valence']])

print("stat: ", stat, ":", "p :", p)
if p > 0.05:
    print("gaussian distribution - fail to reject H0")
else:
    print("not gaussian distribution - reject H0")

    song_df = song_df.drop(['time_signature', 'song_duration_ms'], axis=1)
    song_df.head()

from sklearn.preprocessing import MinMaxScaler

minmaxscaler = MinMaxScaler()
minmaxscaled = minmaxscaler.fit_transform(song_df[['song_popularity', 'acousticness',
                                                   'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                                                   'loudness', 'audio_mode', 'speechiness', 'tempo', 'audio_valence']])
songs_normalized = pd.DataFrame(minmaxscaled, columns=[['song_popularity', 'acousticness',
                                                        'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                                                        'loudness', 'audio_mode', 'speechiness', 'tempo',
                                                        'audio_valence']])

song_df_normalized = song_df.copy()
song_df_normalized.head()

song_df_normalized[['song_popularity', 'acousticness',
                    'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                    'loudness', 'audio_mode', 'speechiness', 'tempo', 'audio_valence']] = songs_normalized
# song_df_normalized.drop(['bpm','nrgy','dnce','dB','live','val','acous','spch','pop'],axis=1,inplace=True)
song_df_normalized.head()

song_features = song_df_normalized.set_index("song_name")
song_features.drop(["artist_name"], axis=1, inplace=True)
song_features.head()

song_features.info()

song_features.tail()

from scipy.sparse import csr_matrix

song_features_csr = csr_matrix(song_features.values)

song_features_csr

song_features.values

from sklearn.neighbors import NearestNeighbors

model_nn = NearestNeighbors(metric='cosine', algorithm='brute')
model_nn.fit(song_features_csr)

temp=song_features.copy()
temp.reset_index(inplace=True)

print(song_features_csr)
def recommend(songsearch):
    songsearch=songsearch.lower()
    song_index=temp.index[temp['song_name'] == songsearch].tolist()[0]
    print(song_index)
    print(song_features.index[song_index])
    distances, indices = model_nn.kneighbors(X=song_features.iloc[song_index, :].values.reshape(1, -1), n_neighbors=6)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print("Recommendation for ", song_features.index[song_index], "are: ")
        else:
            print(i, ": ", song_features.index[indices.flatten()[i]], "| distance= ", distances.flatten()[i])


recommend('By The Way')