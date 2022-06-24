import pandas as pd
processed_data_location =  r'/home/vasanth/jenkins/mlproject/data/processed/data.csv'

df = pd.read_csv(r'/home/vasanth/jenkins/mlproject/data/raw/data.csv',index_col=False)

# daily data saving

df.to_csv(r'/home/vasanth/jenkins/mlproject/data/processed/daily_data.csv', index = False)



# predictions data
target = 'quality'
features=df.columns
features=features.drop('quality')
test = df[features]
print('test frame', test.head())
test.to_csv(processed_data_location,index = False)
