# import statements
import pandas as pd
import glob
from fbprophet import Prophet
import matplotlib.pyplot as plt

# environment variables
# file_input should be the following correspondingly
# overall -> 'data/*/*.json'
# california -> 'data/california/*.json'
# florida -> 'data/florida/*.json'
# michigan -> 'data/michigan/*.json'
# newyork -> 'data/newyork/*.json'
# santaClara -> 'data/santaClara/*.json'
# virginia -> 'data/virginia/*.json'
# washington -> 'data/washington/*.json'

file_input = 'data/*/*.json'

# data ingress
files = glob.glob(file_input)
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

# feature extraction
data_df = data_df[['timestamp', 'd1_congestion_index']]
data_df.columns = ['ds', 'y']

# building a forecasting model based on machine learning
m = Prophet(changepoint_prior_scale=0.01).fit(data_df)
future = m.make_future_dataframe(periods=300, freq='H')
fcst = m.predict(future)

# method to parser file output names
def file_output_name(file_in):
    if file_in.split('/')[1] == '*':
        return 'pages/overall/'
    elif file_in.split('/')[1] == 'california':
        return 'pages/california/'
    elif file_in.split('/')[1] == 'florida':
        return 'pages/florida/'
    elif file_in.split('/')[1] == 'michigan':
        return 'pages/michigan/'
    elif file_in.split('/')[1] == 'newyork':
        return 'pages/newyork/'
    elif file_in.split('/')[1] == 'santaClara':
        return 'pages/santaClara/'
    elif file_in.split('/')[1] == 'virginia':
        return 'pages/virginia/'
    elif file_in.split('/')[1] == 'washington':
        return 'pages/washington/'


file_path = file_output_name(file_input)

# plot the results of the model
fig = m.plot(fcst)
plt.savefig(file_path + 'trend.png')
figN = m.plot_components(fcst)
plt.savefig(file_path + 'subtrends.png')
