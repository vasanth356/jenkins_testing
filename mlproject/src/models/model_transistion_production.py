import yaml
from mlflow.tracking import MlflowClient

# loading the config file
with open ("/home/vasanth/jenkins/mlproject/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# loading the parameters required
exper_name = f"name='{cfg['experiment_details']['name']}'"
client = MlflowClient(tracking_uri = cfg['location']['tracking_uri'] , registry_uri = cfg['location']['registry_uri'])


#getting the versions of the model
versions = []
for mv in client.search_model_versions(exper_name):
 versions.append(dict(mv))
print('all the versions', versions)
latest_version = versions[-1]['version']
old_version = versions[-1]['version']

# changing the latest version to production
client.transition_model_version_stage(
    name=cfg['experiment_details']['name'],
    version=latest_version,
    stage="Production"
)
# changing the latest version to production
client.transition_model_version_stage(
    name=cfg['experiment_details']['name'],
    version=old_version,
    stage="Archived"
)
