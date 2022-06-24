import yaml
import pandas as pd
import joblib
import mlflow
from utils.mlflow_class import *
from mlflow.tracking import MlflowClient

# loading the config file
with open ("/home/vasanth/jenkins/mlproject/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# loading the parameters required
exper_name = f"name='{cfg['experiment_details']['name']}'"
client = MlflowClient(tracking_uri = cfg['location']['tracking_uri'] , registry_uri = cfg['location']['registry_uri'])


df = pd.read_csv(cfg['location']['today_data'])
print(df.columns)
print('test dataframe',df.head())


# getting the latest version
versions = []
for mv in client.search_model_versions(exper_name):
 versions.append(dict(mv))
print('all the versions', versions)
latest_version = versions[-1]['version']
print('latest_version is ', latest_version)
# loading the model from the model registry
model_fetch = GettingModel(cfg['experiment_details']['name'], latest_version)
model_mlflow = model_fetch.model()
output = model_mlflow.predict(df)

# loading the model it to a file
results = pd.DataFrame(output)
results.to_csv(cfg['location']['today_data_predictions'])
