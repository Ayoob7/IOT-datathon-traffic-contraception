# import statements
import pandas as pd
import glob
import gmplot

# data ingress
files = glob.glob('data/*/*.json')
data_df = pd.concat([pd.read_json(file) for file in files], ignore_index=True)

# data cleaning
data_df.drop(['sid', 'has_quality_warning', 'tv_host'], inplace=True, axis=1)

# feature engineering
# engineering the congestion levels and index for the D0 direction of the road
data_df['d0_congestion_level'] = data_df.apply(lambda row: row['d0']['congestion_level'], axis=1)
data_df['d0_congestion_index'] = data_df.apply(lambda row: row['d0']['congestion_index'], axis=1)

# engineering the congestion levels and index for the D0 direction of the road
data_df['d1_congestion_level'] = data_df.apply(lambda row: row['d1']['congestion_level'], axis=1)
data_df['d1_congestion_index'] = data_df.apply(lambda row: row['d1']['congestion_index'], axis=1)

# data cleaning
# necessity to remove "unknown" congestion levels and indexes hence filtering the (-1) values
data_df.drop(['d0', 'd1', 'state', 'timezone', 'name'], inplace=True, axis=1)
data_df = data_df[data_df['d0_congestion_index'] != -1]
data_df = data_df[data_df['d0_congestion_level'] != -1]
data_df = data_df[data_df['d1_congestion_index'] != -1]
data_df = data_df[data_df['d1_congestion_level'] != -1]

bar_index = 5
data_df = data_df[(data_df['d0_congestion_index'].astype(float) > bar_index) & (
        data_df['d1_congestion_index'].astype(float) > bar_index)]

# data map input
data_map = data_df[['latitude', 'longitude']]
gmap = gmplot.GoogleMapPlotter(37.0902, -95.7129, 5)
gmap.scatter(data_map['latitude'], data_map['longitude'], '#FF0000', size=50, marker=True)
gmap.heatmap(data_map['latitude'], data_map['longitude'])
# gmap.apikey = "INSERT YOUR GCP API KEY HERE"
gmap.draw("pages/usa_heatmap.html")

print("Successfully loaded the heat map.")