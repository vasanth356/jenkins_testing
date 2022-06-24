import pandas as pd
import pytest
import os, pathlib
import yaml
from utils.mlflow_class import *

os.chdir( pathlib.Path.cwd())

pytest.main()

# loading the config file
with open ("/home/vasanth/airflow/scripts/mlproject/config.yml", "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
# loading the parameters required
exper_name = f"name='{cfg['experiment_details']['name']}'"
# exper_name = 'random_forest_3'
client = MlflowClient(tracking_uri = cfg['location']['tracking_uri'] , registry_uri = cfg['location']['registry_uri'])
model_name = cfg['model_parameters']['model_type']


EXPECTED_FEATURES = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']

class TestGetUserInput(object):

    def test_data_dimension(self):
        
        """
        This function ensures that the type of data ingested into the sys
        is in the right dimension
        """
        data = pd.read_csv(cfg['location']['today_data_with_target'])
        assert data.ndim == 2

    def test_check_missing_values(self):
        """
        This function test if there is nan values in the test data
        """
        # check that file exists
        data = pd.read_csv(cfg['location']['today_data_with_target'])
        n_nan = np.sum(np.isnan(data.values))
        assert n_nan >= 0

    def test_data_feature_names(self):
        """
        This function test if the ingetsed data have the features in the right order
        """
        data = pd.read_csv(cfg['location']['today_data_with_target'])
        assert list(data.columns) == EXPECTED_FEATURES

    def test_convert_features_type_to_int(self):
        """
        This function test if the ingested data have the datatypes in the right order
        """
        data = pd.read_csv(cfg['location']['today_data_with_target'])
        assert data.dtypes.unique()[0] == 'float64'

    def test_non_empty(self):
        """ensures that there is more than 1000 row of data"""
        data = pd.read_csv(cfg['location']['today_data_with_target'])
        assert data.size > 1000

    def test_mlflow_predictions(self):
        """ This function test if we are able to predict the outputs by using mlflow """
        data = pd.read_csv(cfg['location']['today_data_with_target'])
        data.drop(['quality'], axis = 1, inplace = True)

        # getting the latest version from mlflow registry
        versions = []
        for mv in client.search_model_versions(exper_name):
         versions.append(dict(mv))
        latest_version = versions[-1]['version']

        # loading the model from the model registry
        model_fetch = GettingModel(cfg['experiment_details']['name'], latest_version)
        model_mlflow = model_fetch.model()
        output = model_mlflow.predict(data)
        assert output.size != 0
