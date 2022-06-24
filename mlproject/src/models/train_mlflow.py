# ### simple implementation of the mlflow class
import os
import pandas as pd
import yaml
from sklearn.ensemble import  *
from sklearn.linear_model import *
from utils.mlflow_class import *

# loading the config file
with open ("/home/vasanth/jenkins/mlproject/config.yml", "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
# loading the parameters required
exper_name = f"name='{cfg['experiment_details']['name']}'"
client = MlflowClient(tracking_uri = cfg['location']['tracking_uri'] , registry_uri = cfg['location']['registry_uri'])
model_name = cfg['model_parameters']['model_type']
print(type(model_name))

df = pd.read_csv(cfg['location']['today_data_with_target'])

from sklearn.ensemble import  *
from sklearn.linear_model import *

#preprocessing
target = cfg['model_parameters']['target']
features = df.columns
features=features.drop(target)

data = [df[features], df[target]]

test = df.iloc[0:1000]
test_x = test[features]
test_y = test[target]

test_data = [test_x, test_y]
model = RandomForestRegressor()

model_parameters =  cfg['model_parameters']['parameters']

print('model parameters for the model is ', model_parameters)

# creating the mlflow experiment object
ml_testing = MlflowExperiment(data,test_data, model,model_parameters,cfg['experiment_details']['name'], cfg['model_parameters']['problem_type'])
ml_testing.mlflow_runs()

print('experiment_name passed', cfg['experiment_details']['name'])


# prediction from the best trained model
# getting the latest version
versions = []
for mv in client.search_model_versions(exper_name):
 versions.append(dict(mv))
print('all the versions before registering the model', versions)


# selecting the best run from the experiment
run  = SelectBestRun(cfg['experiment_details']['name'])
print('experiment results', run.experiment_results)
print('getting the run id', run.best_run_id())
run.register_model(cfg['experiment_details']['name'])


# prediction from the best trained model
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
output = model_mlflow.predict(df[features])
results = pd.DataFrame(output,columns =['prediction'] )
df = pd.concat([df,results], axis=1)
df.to_csv(cfg['location']['trained_model_predictions'],index = False)
