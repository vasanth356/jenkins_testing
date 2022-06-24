import yaml
import pandas as pd
import os
import joblib
from utils.mlflow_class import *
from mlflow.tracking import MlflowClient

# loading the config file
with open ("/home/vasanth/jenkins/mlproject/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# loading the parameters required
exper_name = f"name='{cfg['experiment_details']['name']}'"
client = MlflowClient(tracking_uri = cfg['location']['tracking_uri'] , registry_uri = cfg['location']['registry_uri'])


df = pd.read_csv(cfg['location']['yesterday_data'])
target = cfg['model_parameters']['target']
features=df.columns
features=features.drop(target)
yesterday_data = df[features]

print('current working directory', os.getcwd())

# loading the model it to a file
versions = []

for mv in client.search_model_versions(exper_name):
 versions.append(dict(mv))
print('all the versions', versions)
latest_version = versions[-1]['version']
print('latest_version is ', latest_version)
# loading the model from the model registry
model_fetch = GettingModel(cfg['experiment_details']['name'], latest_version)
model_mlflow = model_fetch.model()
output = model_mlflow.predict(yesterday_data)
results = pd.DataFrame(output,columns =['prediction'] )



df = pd.concat([df,results], axis=1)
df.to_csv(cfg['location']['yesterday_data_predictions'],index = False)
