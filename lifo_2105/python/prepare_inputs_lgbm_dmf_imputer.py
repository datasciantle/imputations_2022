#!/usr/bin/python
# docker run -it -v /tmp:/tmp  --rm 126345656468.dkr.ecr.us-east-1.amazonaws.com/quantumplethodon:rkumar-tf-001 bash 
# ssh ubuntu@10.215.10.61
# import tkinter
# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt
# [LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN

"""
Work around for ancient docker containers: install lightgbm on the fly
"""
import sys
import subprocess
import pkg_resources
required = {'lightgbm'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

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

# Numerical / data modules
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score,accuracy_score,roc_auc_score,matthews_corrcoef,log_loss
from sklearn.model_selection import train_test_split
from sklearn import feature_selection

# Machine learning modules
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, pyll, space_eval
import xgboost as xgb
import lightgbm as lgb

## Defaults for prototyping in shell
base_s3_path = 's3://datasci-scantle/unified_ncs/experiments/2105'
model_version = 'lifo_2105'
s3_complete_response_file = 's3://datasci-scantle/unified_ncs/2105/lifo/inputs/complete_responses_libsvm_1103SS_9551MS_10654TotalAttributes'
s3_complete_taxonomy_file = 's3://datasci-scantle/unified_ncs/2105/lifo/metadata/complete_taxonomy_indexed'
s3_latent_distribution_file = 's3://datasci-scantle/unified_ncs/2105/dmf_latent_distribution_predictions/self_supervised_svkey_embedding_tabnet_dmf_tf_efc216265f32_epoch_19'

## (1)  SETUP
print('I am starting LightGBM model development.')
## make the homedir /tmp
os.environ["HOME"] = "/tmp"

# Set various parameters including arg.parse
# Note: please pay attention to the last slash "/" on paths because it can interfere with pyarrow...
start_time = int(round(time.time() * 1000))
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

parser.add_argument('--features_column_name', default='complete_response_libsvm', type=str,
                    help='The name of the column that contains the features for training. (Future facing to facilitate experimentation.)')
## response_file
parser.add_argument('--s3_complete_response_file', default=s3_complete_response_file, type=str,
                    help='The s3 location of the panelist response file.')

parser.add_argument('--s3_complete_taxonomy_file', default=s3_complete_taxonomy_file, type=str,
                    help='The s3 location of the indexed attribute taxonomy.')

parser.add_argument('--s3_latent_distribution_file', default=s3_latent_distribution_file, type=str,
                    help='The s3 location of the DMF latent distribution predictions.')
## model parameters
parser.add_argument('--iteration', default=0, type=int,
                    help='The iteration of imputation you are pursuing.')

args = parser.parse_args()


#####################################################################################
###########                      ETL BLOCK                                ###########
#####################################################################################
working_directory = os.getenv('HOME') + '/' + args.model_version
os.makedirs(working_directory, exist_ok=True)
os.makedirs(working_directory + '/inputs/complete_response', exist_ok=True)
os.makedirs(working_directory + '/inputs/labels', exist_ok=True)
# Here we enumerate the output files
local_complete_response_file_libsvm = os.getenv('HOME') + '/' + args.model_version + '/inputs/complete_response/complete_response_libsvm'
local_complete_response_file_uids = os.getenv('HOME') + '/' + args.model_version + '/inputs/complete_response/complete_response_uids'
local_pickle_file = os.getenv('HOME') + '/' + args.model_version + '/inputs/local_pickle_data.pickle'
s3_pickle_file = args.base_s3_path + '/python_inputs/iteration_' + str(args.iteration) + '.pickle' 


print('I am starting ETL and other preprocessing stages.')
# Here we collect model information.
s3 = s3fs.S3FileSystem()
features_column = args.features_column_name
n_single_select_features = int(args.s3_complete_response_file.split('SS_')[0].split('_')[-1])
n_multi_select_features = int(args.s3_complete_response_file.split('MS_')[0].split('_')[-1])
n_total_features = int(args.s3_complete_response_file.split('TotalAttributes')[0].split('_')[-1])

# if not (!os.path.isfile(local_pickle_file)):
if not (s3.exists(s3_pickle_file)):
    print('I could not find the local_complete_response files: ' + local_complete_response_file_libsvm +' and ' + local_complete_response_file_uids + '. Downloading now.')
    complete_taxonomy_df = pq.ParquetDataset(args.s3_complete_taxonomy_file, filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()
    model_plan_df = pq.ParquetDataset(args.s3_complete_taxonomy_file.replace('complete_taxonomy_indexed','model_plan'), filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()
    distributions_df = pq.ParquetDataset(args.s3_complete_taxonomy_file.replace('complete_taxonomy_indexed','distributions'), filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()
    local_complete_response = pq.ParquetDataset(args.s3_complete_response_file, filesystem=s3, metadata_nthreads=10).read_pandas(columns=['uid','wavekey','set_uid', features_column]).to_pandas()
    latent_distribution_files = []
    for folder in ['train','test','validation']:
        files = s3.ls(args.s3_latent_distribution_file + '/' + folder)
        for file in files:
            latent_distribution_files.append('s3://' + file)
    
    latent_distribution_df = pd.concat((pd.read_csv(f) for f in latent_distribution_files))
    local_complete_response = local_complete_response.merge(latent_distribution_df, left_on=['uid'], right_on=['uid'])
    latent_distribution_df = local_complete_response[[x for x in local_complete_response.columns if ("uid" in x  or "ld" in x)]]
    local_complete_response['libsvm'] = '0.0  ' + local_complete_response[features_column]
    local_complete_response.to_csv(local_complete_response_file_libsvm, header=False, index=False, columns=['libsvm'])
    X,y = load_svmlight_file(local_complete_response_file_libsvm, n_features = n_total_features)
    data_dict = {'n_single_select_features':n_single_select_features,'n_multi_select_features':n_multi_select_features,'n_total_features':n_total_features,'complete_taxonomy_df':complete_taxonomy_df, 'model_plan_df':model_plan_df, 'distributions_df':distributions_df, 'latent_distribution_df':latent_distribution_df, 'complete_responses_sparse':X}
    pikd = open(local_pickle_file, 'wb')
    pickle.dump(data_dict, pikd)
    # pickle.dump(X, pikd)
    pikd.close()
    sync_cmd = 'aws s3 cp ' + local_pickle_file + ' ' + s3_pickle_file
    os.system(sync_cmd)
