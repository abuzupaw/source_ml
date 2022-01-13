# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:46:00 2022

@author: acer
"""

import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap

import pandas as pd
import numpy as np





df_tps = pd.read_csv("C:/Users/acer/Documents/data_tps.csv", sep=',')

df_tps_filter = df_tps[['WILAYAH', 'NAMA_TPS','Latitude', 'Longitude', 'MASUK', 'KE_TPA']]


# koordinat = df_tps_filter.iloc[0, [2,3]].to_list()
# nama = df_tps_filter.iloc[0, [1]].values.astype(str)
# nama = str(nama)
# folium.Marker(koordinat, popup = nama ).add_to(my_map)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_tps_filter['ID_WILAYAH'] = le.fit_transform(df_tps_filter['WILAYAH'])

#Define coordinates of where we want to center our map
boulder_coords = [-6.85544, 107.592]

#Create the map
my_map = folium.Map(location = boulder_coords, zoom_start = 13)
my_heatmap = folium.Map(location = boulder_coords, zoom_start = 13)

# df_tps_filter.groupby('ID_WILAYAH').size()

# df_tps_filter.iloc[0, [1]].astype(str).to_string()
#‘red’, ‘blue’, ‘green’, ‘purple’, ‘orange’, ‘darkred’,

#’lightred’, ‘beige’, ‘darkblue’, ‘darkgreen’, ‘cadetblue’, ‘darkpurple’, ‘white’, ‘pink’, ‘lightblue’, ‘lightgreen’, ‘gray’, ‘black’, ‘lightgray’]



for i in range (0, len(df_tps_filter)):
    koordinat = df_tps_filter.iloc[i, [2,3]].to_list()
    nama = df_tps_filter.iloc[i, [1]].astype(str).to_string()
    if(df_tps_filter.iloc[i,6]==0):
        folium.Marker(koordinat, popup = nama,icon=folium.Icon(color='green')).add_to(my_map)
    elif(df_tps_filter.iloc[i,6]==1):
        folium.Marker(koordinat, popup = nama,icon=folium.Icon(color='blue')).add_to(my_map)
    elif(df_tps_filter.iloc[i,6]==2):
        folium.Marker(koordinat, popup = nama,icon=folium.Icon(color='red')).add_to(my_map)
    elif(df_tps_filter.iloc[i,6]==3):
        folium.Marker(koordinat, popup = nama,icon=folium.Icon(color='green',icon_color='black')).add_to(my_map)
    elif(df_tps_filter.iloc[i,6]==4):
        folium.Marker(koordinat, popup = nama,icon=folium.Icon(color='blue',icon_color='black')).add_to(my_map)
    elif(df_tps_filter.iloc[i,6]==5):
        folium.Marker(koordinat, popup = nama,icon=folium.Icon(color='red',icon_color='black')).add_to(my_map)
    


my_map.save("C:/Users/acer/Documents/latihan_leaflet/mymap.html")


################################################ 
############# KLASTERING
################################################
from sklearn.cluster import KMeans
jmlcluster = 3 

modelKmean = KMeans(n_clusters=jmlcluster).fit(df_tps_filter[['MASUK','KE_TPA']])

df_tps_filter['hasilKmean'] = pd.DataFrame(modelKmean.labels_)