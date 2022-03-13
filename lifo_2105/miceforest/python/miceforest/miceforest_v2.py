#!/usr/bin/python
# docker rmi $(docker images -q) --force
# $(aws ecr get-login --no-include-email --region us-east-1)
# docker run -it -v /tmp:/tmp  --rm 126345656468.dkr.ecr.us-east-1.amazonaws.com/quantumplethodon:rkumar-tf-001 bash 
# ssh ubuntu@10.215.10.61
# import tkinter
# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt

"""
Work around for ancient docker containers: install lightgbm on the fly
"""
import sys
import subprocess
import pkg_resources
required = {'lightgbm','dask','miceforest'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

# subprocess.check_call([python, '-m', 'pip', 'install', '--upgrade', {'s3fs'}], stdout=subprocess.DEVNULL)
# Environmental modules

import argparse
import boto3
from botocore.client import Config
import gc
import multiprocessing
import os
import pickle
import pyarrow.parquet as pq
import random
import s3fs
import shutil
import sys
import time
import uuid
from functools import partial
from itertools import compress
import multiprocessing
# Numerical / data modules
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn import feature_selection
from sklearn import metrics
from scipy import optimize
from scipy import special
from sklearn.preprocessing import LabelBinarizer



# Machine learning modules
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe, pyll, space_eval
from hyperopt.pyll import scope
import lightgbm as lgb
import miceforest as mf

# Defaults for prototyping in shell

# base_s3_path = 's3://datasci-scantle/unified_ncs/experiments/2105_miceforest'
# model_version = 'lifo_2105'
# s3_attribute_matrix = 's3://datasci-scantle/unified_ncs/2105/inputs/attribute_matrix'
# s3_complete_taxonomy_file = 's3://datasci-scantle/unified_ncs/2105/lifo/metadata/complete_taxonomy_indexed'
# s3_distributions = 's3://datasci-scantle/unified_ncs/2105/lifo/metadata/distributions'
# s3_feature_input_to_round = 's3://datasci-scantle/unified_ncs/experiments/2105_miceforest/iteration_0/inputs/core_and_dmf'
# s3_valid_respondents = 's3://datasci-scantle/unified_ncs/2105/bsm/data/validRespondents'
# model = '381473'
# model = '261625_261632'


base_s3_path = 'fake'
model_version = 'fake'
s3_attribute_matrix = 'fake'
s3_complete_taxonomy_file = 'fake'
s3_distributions = 'fake'
s3_feature_input_to_round = 'fake'
s3_valid_respondents = 'fake'
model = 'fake'




## (1)  SETUP
print('I am starting LightGBM model development.')
## make the homedir /tmp
os.environ["HOME"] = "/tmp"

# Set various parameters including arg.parse
# Note: please pay attention to the last slash "/" on paths because it can interfere with pyarrow...
start_time = time.time()
model_uuid = 'lgbmpy_' + uuid.uuid1().hex[:12]
print("This model's UID is: " + model_uuid)
config = Config(connect_timeout=5, retries={'max_attempts': 10})

parser = argparse.ArgumentParser(description='Gradient Boosting - Training with Hyperopt Optimization')

parser.add_argument('--base_s3_path', default=base_s3_path, type=str,
                    help='The s3 location where everything will write.')

parser.add_argument('--model_version', default=model_version, type=str,
                    help='The name of the model version.')

parser.add_argument('--iteration', default=0, type=int,
                    help='The iteration of the imputation.')
## response_file
parser.add_argument('--s3_attribute_matrix', default=s3_attribute_matrix, type=str,
                    help='The s3 location of the attribute BSM.')

parser.add_argument('--s3_complete_taxonomy_file', default=s3_complete_taxonomy_file, type=str,
                    help='The s3 location of the indexed attribute taxonomy.')

parser.add_argument('--s3_distributions', default=s3_distributions, type=str,
                    help='The s3 location of the distributions file.')

parser.add_argument('--s3_feature_input_to_round', default=s3_feature_input_to_round, type=str,
                    help='The s3 location of the features to use for generating imputations.')

parser.add_argument('--s3_valid_respondents', default=s3_valid_respondents, type=str,
                    help='The s3 location of the valid respondents table.')
## model parameters
parser.add_argument('--model', default=model, type=str,
                    help='The particular model you are training, written as sakey_svkey_behaviorType.')

parser.add_argument('--lifo_round', default=0, type=int,
                    help='The display logic round from which you may draw random covariates.')

parser.add_argument('--n_random_attributes', default=5, type=int,
                    help='The number of random attributes to use in the MICE procedure.')

parser.add_argument('--mice_iterations', default=5, type=int,
                    help='The number of MICE iterations.')

parser.add_argument('--delete_local', default=True, type=bool,
                    help='Whether or not to delete local output after pushing to s3, important to avoid blowing out limited memory of training instances.')

parser.add_argument('--redo', default=False, type=bool,
                    help='Whether or not to re-run a model that previously completed')

args = parser.parse_args()

#####################################################################################
###########                      ETL BLOCK                                ###########
#####################################################################################
print('I am starting ETL and other preprocessing stages.')
# Here we collect model information.
s3 = s3fs.S3FileSystem()
model = args.model
multi = "_" in model
this_iteration = [x.split('_')[-1] for x in args.s3_feature_input_to_round.split('/') if 'iteration' in x][0]
was_completed = s3.isfile(args.base_s3_path + '/iteration_' + str(this_iteration) + '/miceforest/completeness/' + args.model)

if((was_completed)& (args.redo is False)):
    print('I already made this model and you told me not to redo it.')
    quit()

working_directory = os.getenv('HOME') + '/' + args.model_version
output_directory = working_directory + '/output/' + args.model
os.makedirs(working_directory, exist_ok=True)
os.makedirs(working_directory + '/inputs', exist_ok=True)
os.makedirs(output_directory, exist_ok=True)


# Here we enumerate the output files
local_imputations_file = working_directory + '/inputs/local_imputation_data'
local_dmf_file = working_directory + '/inputs/local_dmf_data'
local_focal_predictions_file = output_directory + '/' + model + '_focal_predictions'
local_total_predictions_file = output_directory + '/' + model + '_total_predictions'
local_feature_importance_file = output_directory + '/' + model + '_featureImportance'
local_distribution_file = output_directory + '/' + model + '_distributions'

print('I am starting ETL and other preprocessing stages.')
# Here we collect model information.
s3 = s3fs.S3FileSystem()

feature_input_files = ['s3://' + x for x in s3.ls(args.s3_feature_input_to_round) if not '_SUCCESS' in x]
features = dd.read_csv(feature_input_files).compute()

valid_respondents = pq.ParquetDataset(args.s3_valid_respondents, filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()[['uid','wavekey','respondentweight']]
valid_respondents['weight'] = valid_respondents['respondentweight'].values.astype(np.float64)

complete_taxonomy_df = pq.ParquetDataset(args.s3_complete_taxonomy_file, filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()
distributions_df = pq.ParquetDataset(args.s3_distributions, filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()
taxonomy = distributions_df.merge(complete_taxonomy_df[['sakey','svkey','attribute','value','proposed_round']], left_on=['sakey','svkey'], right_on=['sakey','svkey'])
taxonomy.loc[taxonomy.value.isnull(),'value'] = 2.
taxonomy['attribute_value'] = taxonomy['value'] - 1.
taxonomy['attribute_value'] = taxonomy['attribute_value'].astype('int')
taxonomy = taxonomy.drop(columns=['value'])

possible_round_attributes = [x for x in taxonomy.loc[(taxonomy.proposed_round == args.lifo_round) & (taxonomy.needs_imputation), 'attribute'].drop_duplicates().values]
possible_round_attributes = [x for x in possible_round_attributes if not x in features.columns]
imputation_columns = list(np.append(model,np.sort(np.random.choice(np.array(possible_round_attributes), args.n_random_attributes, replace=False))))
this_taxonomy = taxonomy.loc[taxonomy.attribute.isin(imputation_columns)]

print(this_taxonomy)

for model_col in imputation_columns:
    model_is_multi = '_' in model_col
    labels = pq.ParquetDataset(args.s3_attribute_matrix + '/attribute2=' + model_col, filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()[['uid','set_uid','label']]
    if not model_is_multi:
        labels.loc[(labels.label.notnull()),'label'] = labels.loc[(labels.label.notnull()),'label'] -1
    labels[model_col] = labels['label']
    labels = labels[['uid','set_uid', model_col]]
    if model_col == imputation_columns[0]:
        imputation = labels
    else:
        imputation = imputation.merge(labels, left_on=['uid','set_uid'], right_on=['uid','set_uid'], how='outer')



X_all = imputation.merge(features, left_on=['uid','set_uid'], right_on=['uid','set_uid'], how='outer')
for x in [x for x in X_all.columns if ((not x in ['uid','set_uid','wavekey']) and ('ld_' not in x))]:
    X_all.loc[:,x] = X_all[x].astype('category') 

X = X_all[[x for x in X_all.columns if (not x in ['uid','set_uid','wavekey'])]]

print(X)

imp_kernel = mf.ImputationKernel(
    X,
    datasets=1,
    save_all_iterations=True,
    random_state=42,
    train_nonmissing=False,
    variable_schema=imputation_columns
    )

imp_kernel.mice(args.mice_iterations,verbose=True)

compdata = imp_kernel.complete_data(dataset=0)[imputation_columns]
compdata = pd.concat([X_all[['uid','set_uid','wavekey']], compdata], axis=1)
compdata = compdata.merge(valid_respondents[['uid','weight']], left_on=['uid'], right_on=['uid'])
waves = list(np.sort(np.unique(compdata['wavekey'])))

for idx in imputation_columns:
    wave_distribution_dict = dict()
    for wave in waves:
        x = compdata.loc[compdata.wavekey==wave][['weight',idx]]
        wave_distribution_dict[str(wave)] = np.bincount(x[idx], weights=x.weight) / sum(x.weight)
    wave_distribution_df = pd.DataFrame(wave_distribution_dict)
    wave_distribution_df['imputed_distribution'] = wave_distribution_df.values.mean(axis=1)
    wave_distribution_df['attribute'] = idx
    wave_distribution_df['attribute_value'] = wave_distribution_df.index
    if idx == imputation_columns[0]:
        distribution_results_df = wave_distribution_df
    else:
        distribution_results_df = pd.concat([distribution_results_df,wave_distribution_df])

distribution_results_df = this_taxonomy.merge(distribution_results_df, left_on=['attribute','attribute_value'], right_on=['attribute','attribute_value'])
distribution_results_df['target_attribute'] = args.model
print(distribution_results_df)

target_predictions = pd.melt(compdata, id_vars=['uid','set_uid','wavekey'], value_vars=imputation_columns, var_name='attribute', value_name='selected_value')
target_labels = pd.melt(X_all, id_vars=['uid','set_uid','wavekey'], value_vars=imputation_columns, var_name='attribute', value_name='label')
target_predictions['attribute_model'] = args.model
target_predictions = target_labels.merge(target_predictions, left_on=['uid','set_uid','wavekey','attribute'], right_on=['uid','set_uid','wavekey','attribute'])

feature_importance =  pd.DataFrame(imp_kernel.get_feature_importance(dataset=0), columns=[x for x in X_all.columns if not x in ['uid','set_uid','wavekey']])
feature_importance['attribute'] = imputation_columns
feature_importance['attribute_model'] = args.model
feature_importance = pd.melt(feature_importance, id_vars=['attribute_model','attribute'], value_vars=[x for x in feature_importance.columns if not x in ['attribute', 'attribute_model']], var_name='feature', value_name='feature_importance')

# Time to save everything
target_predictions.loc[target_predictions.attribute == target_predictions.attribute_model].to_csv(local_focal_predictions_file, columns=['uid','set_uid','wavekey','attribute_model','attribute','label','selected_value'], index=False)
target_predictions.to_csv(local_total_predictions_file, columns=['uid','set_uid','wavekey','attribute_model','attribute','label','selected_value'], index=False)
feature_importance.to_csv(local_feature_importance_file, index=False)
distribution_results_df.to_csv(local_distribution_file, index=False)

final_sync_cmd = 'aws s3 sync ' + output_directory + ' ' + args.base_s3_path + '/iteration_' + str(this_iteration) + '/miceforest/results/' + args.model
os.system(final_sync_cmd)

finish_time = time.time()
times_df = pd.DataFrame([{'model':args.model, 'iteration':this_iteration, 'start_time':start_time, 'finish_time':finish_time, 'total_time':finish_time-start_time}])
times_df.to_csv(args.base_s3_path + '/iteration_' + str(this_iteration) + '/miceforest/completeness/' + args.model, index=False)

if args.delete_local:
    print('Cleaning up local')
    shutil.rmtree(output_directory)

