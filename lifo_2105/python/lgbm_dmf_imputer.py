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
from itertools import compress

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
from hyperopt.pyll import scope
import xgboost as xgb
import lightgbm as lgb

## Defaults for prototyping in shell
base_s3_path = 's3://datasci-scantle/unified_ncs/experiments/2105'
model_version = 'lifo_2105'
s3_pickle_file = 's3://datasci-scantle/unified_ncs/experiments/2105/python_inputs/iteration_0.pickle'
model = '128997'

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
## response_file
parser.add_argument('--s3_pickle_file', default=s3_pickle_file, type=str,
                    help='The s3 location of the python dataset environment file.')
## model parameters
parser.add_argument('--model', default=model, type=str,
                    help='The particular model you are training, written as sakey_svkey_behaviorType.')

parser.add_argument('--lgbm_max_iterations', default=200, type=int,
                    help='The maximum number of trees (recall that early stopping is also used).')

parser.add_argument('--hyperopt_max_evals', default=50, type=int,
                    help='The maximum number of evaluations for hyperparameter optimization in hyperopt.')

parser.add_argument('--delete_local', default=True, type=bool,
                    help='Whether or not to delete local output after pushing to s3, important to avoid blowing out limited memory of training instances.')

parser.add_argument('--verbose', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='Whether or not to print all output from training routines.  You probably should only use \'True\' for debugging.')

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


print('I am starting ETL and other preprocessing stages.')
# Here we collect model information.
s3 = s3fs.S3FileSystem()

if not (os.path.isfile(local_pickle_file)):
    print('I could not find the s3_pickle_file file: ' + s3_pickle_file + '. Downloading now.')
    os.system('aws s3 cp ' + args.s3_pickle_file + ' ' + local_pickle_file)

pikd = open(local_pickle_file, 'rb')
data_dict = pickle.load(pikd)
pikd.close()


complete_taxonomy = data_dict['complete_taxonomy_df']
X = data_dict['complete_responses_sparse']
latent_distribution_df = data_dict['latent_distribution_df']
distributions_df = data_dict['distributions_df']
model_plan_df = data_dict['model_plan_df']

del data_dict
gc.collect()

# exclude age, age/gender, state, model
exclude_attributes = ['129469','129406','129378', model]
mostly_complete_level_0 = model_plan_df.loc[((model_plan_df['n_respondents'] > X.shape[0] * 0.8) & (model_plan_df['proposed_round']==0))]
include_these_attributes = complete_taxonomy.loc[((complete_taxonomy.svkey.isin(mostly_complete_level_0['svkey'].values)) & (~complete_taxonomy.attribute.isin(exclude_attributes)))][['attribute','final_index']].drop_duplicates().sort_values(by='final_index')

raw_attributes = list(include_these_attributes['attribute'].values)

label_index = complete_taxonomy.loc[(complete_taxonomy['attribute']==args.model)]['final_index'].values[0]
y_initial = np.array(X[:,label_index].todense()).flatten()
labeled_indices = y_initial != 0
unlabeled_indices = y_initial == 0
latent_distribution_df['is_labeled'] = 0
latent_distribution_df.loc[labeled_indices,'is_labeled'] = 1
latent_distribution_df['label'] = y_initial - 1
latent_distribution_df = pd.concat([latent_distribution_df, pd.DataFrame(X[:,include_these_attributes['final_index'].values].todense(), columns=raw_attributes).replace(0.0, np.nan) - 1], axis=1)

# Need to do some feature selection to drop things that are perfectly correlated.
latent_distribution_columns = [x for x in latent_distribution_df.columns if "ld_" in x]
all_feature_columns = latent_distribution_columns + raw_attributes
train_set_indices = ((latent_distribution_df['set_uid']=='train') & (latent_distribution_df['is_labeled']==1)).values
test_set_indices = ((latent_distribution_df['set_uid']=='test') & (latent_distribution_df['is_labeled']==1)).values
validation_set_indices = ((latent_distribution_df['set_uid']=='validation') & (latent_distribution_df['is_labeled']==1)).values


X_train = latent_distribution_df[train_set_indices][all_feature_columns]
y_train = latent_distribution_df[train_set_indices]['label'].values
X_test = latent_distribution_df[test_set_indices][all_feature_columns]
y_test = latent_distribution_df[test_set_indices]['label'].values




num_class = len(np.unique(latent_distribution_df[labeled_indices]['label'].values))
objective_function = 'binary' if num_class < 3 else 'multiclassova'
boolean_options = [True,False]

space_lgb = {
    # lightGBM parameters
    # Invariant parameter settings
    'zero_as_missing': False,
    'objective': objective_function, # conditional that depends on whether the model is multi or single select.
    'num_class': num_class,
    # 'is_unbalance': hp.choice('is_unbalance', boolean_options),
    'boosting': 'goss', # goss is the gradient one side sampling lgbm is known for
    'linear_tree': False,
    'tree_learner': 'serial',
    'num_threads': multiprocessing.cpu_count(),
    'device_type': 'cpu',
    'deterministic': False,
    'seed': 42,
    'num_iterations': args.lgbm_max_iterations, # I like the idea of fixing the max trees but allowing for early stopping
    'early_stopping_rounds': 30,
    'feature_pre_filter': True,
    'force_row_wise': True, # There's about 10-15sec of overhead
    # End invariant parameters
    ##########################
    # These are the parameter settings hyperopt will explore.
    'scale_pos_weight': hp.uniform('scale_pos_weight', 0.0001, 1.0)**-1,
    # 'focal_alpha': hp.uniform('focal_alpha', 0.0001, 1.0)**-1,
    # 'focal_gamma': hp.uniform('focal_gamma', 0., 10.),
    'learning_rate': hp.uniform('eta', 0.0001, 0.5),
    'max_delta_step': hp.uniform('max_delta_step', -1., 30.),
    'num_leaves': scope.int(hp.quniform('num_leaves', 2., 4096., 2.)),
    'max_depth': scope.int(hp.quniform('max_depth', 3., 11., 1.)),
    'max_bin': scope.int(hp.quniform('max_bin', 2., 256., 1.)),
    'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 1., 1024., 1.)),
    'min_sum_hessian_in_leaf': hp.lognormal('min_sum_hessian_in_leaf', 3., 3.),
    'min_gain_to_split': hp.lognormal('min_gain_to_split', 3., 3.),
    'bagging_fraction': hp.uniform('bagging_fraction', 0., 1.),
    'feature_fraction': hp.uniform('feature_fraction', 0., 1.),
    'extra_trees': hp.choice('extra_trees', boolean_options),
    'lambda_l1': hp.uniform('lambda_l1', 1e-10, 20.),
    'lambda_l2': hp.uniform('lambda_l2', 1e-10, 20.),
    'top_rate': hp.uniform('top_rate', 0., 1.),
    'other_rate': hp.uniform('other_rate', 0., 1.),
    'cat_l2': hp.uniform('cat_l2', 1e-10, 3.),
    'cat_smooth': hp.uniform('cat_smooth', 1e-10, 30.)
}

space_features = dict()
for f in latent_distribution_columns:
     space_features.setdefault(f, hp.choice(f, boolean_options))

space_weights = dict()
for svk in np.arange(num_class):
    # space_weights.setdefault('class_label' + str(svk + 1), hp.uniform('class_label' + str(svk + 1), 1e-10, 1.)) ** -1
    space_weights.setdefault(str(svk), hp.uniform(str(svk), 1e-10, 1.)) ** -1

space_lgb.update(space_features)
# space_lgb.update(space_weights)
# space_lgb['class_weight'] = space_weights

# from scipy.misc import derivative

# def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
#   a,g = alpha, gamma
#   y_true = dtrain.label
#   def fl(x,t):
#     p = 1/(1+np.exp(-x))
#     return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
#   partial_fl = lambda x: fl(x, y_true)
#   grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
#   hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
#   return grad, hess

# def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
#   a,g = alpha, gamma
#   y_true = dtrain.label
#   p = 1/(1+np.exp(-y_pred))
#   loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
#   # (eval_name, eval_result, is_higher_better)
#   return 'focal_loss', np.mean(loss), False




def score_impute_cv(impute_params, X_train, y_train):
    print("Training GOSS LGBM with params: ")
    # print(impute_params)
    # sample_weights = np.array([impute_params[str(int(x))] for x in y_train])
    # print(sample_weights)
    # sample_weights /= sample_weights.max()
    # [impute_params.pop(str(int(x))) for x in np.arange(num_class)]
    keep_features = [x for x in latent_distribution_columns if impute_params[x] is True] + raw_attributes
    impute_params['other_rate'] = impute_params['other_rate'] * (1. - impute_params['top_rate'])
    [impute_params.pop(x) for x in latent_distribution_columns]
    # # alpha = impute_params.pop('focal_alpha')
    # # gamma = impute_params.pop('focal_gamma')
    # focal_loss = lambda x,y: focal_loss_lgb(x,y, alpha, gamma)
    print(impute_params)
    train_data = lgb.Dataset(data=X_train[keep_features], label=y_train, feature_name=keep_features, categorical_feature=raw_attributes, free_raw_data=True, params={'verbose':-1})
    print('The parameters for this iteration are:')
    cv_res = lgb.cv(impute_params, train_data, nfold=5, stratified=True, shuffle=True, verbose_eval=False, return_cvbooster=True)
    crossval_mean = min(cv_res['score-mean']) if num_class==2 else min(cv_res['multi_logloss-mean'])
    crossval_stdv =  list(compress(cv_res['score-stdv'], list(cv_res['score-mean'] == min(cv_res['score-mean']))))[0] if num_class==2 else list(compress(cv_res['multi_logloss-stdv'], list(cv_res['multi_logloss-mean'] == min(cv_res['multi_logloss-mean']))))[0]
    crossval_var = crossval_stdv**2
    best_iteration = np.argmin(cv_res[objective_function + '-mean']) if num_class==2 else np.argmin(cv_res['multi_logloss-mean'])
    return {'loss': crossval_mean, 'true_loss_variance':crossval_var, 'status': STATUS_OK, 'best_iteration': best_iteration,'crossval_mean': crossval_mean}

trials = Trials()
fmin_this_treatment = partial(score_impute_cv, X_train=X_train, y_train=y_train)
best = fmin(fmin_this_treatment,  space=space_lgb, algo=tpe.suggest, max_evals=50, trials=trials, show_progressbar=True)

space_columns = list(space_lgb.keys())   
searched_space = pd.DataFrame(columns=['hyperopt_iteration'] + space_columns + ['loss'])
for idx, trial in enumerate(trials.trials):
    row = [idx]
    translated_eval = space_eval(space_lgb, {k: v[0] for k, v in trial['misc']['vals'].items()})
    for k in space_columns:
        row.append(translated_eval[k])
    row.append(trial['result']['loss'])
    searched_space.loc[idx] = row

searched_space['model_column'] = model_column
searched_space.to_csv(local_hyperopt_file, index=False)

print('I am training the final model.')
params = space_eval(space_lgb, trials.argmin)
params.update(seed=47)
keep_features = [x for x in latent_distribution_columns if params[x] is True]  + raw_attributes
params['other_rate'] = params['other_rate'] * (1. - params['top_rate'])
[params.pop(x) for x in latent_distribution_columns]

train_weight = np.array([params[str(int(x))] for x in y_train])
train_weight /= train_weight.max()
test_weight = np.array([params[str(int(x))] for x in y_test])
test_weight /= test_weight.max()

train_data = lgb.Dataset(data=X_train[keep_features], label=y_train, weight=train_weight, feature_name=keep_features, categorical_feature=raw_attributes, free_raw_data=True, params={'verbose':-1})
test_data = lgb.Dataset(X_test[keep_features], label=y_test, weight=test_weight, feature_name=keep_features, categorical_feature=raw_attributes, free_raw_data=True, params={'verbose':-1})
bst = lgb.train(params, train_data, valid_sets=[test_data])


these_results = latent_distribution_df[['uid','set_uid','is_labeled','label']]
predictions = bst.predict(latent_distribution_df[keep_features],num_iteration=bst.best_iteration)
these_results = pd.concat([these_results, pd.DataFrame(predictions)], axis=1)
these_results['max_prediction'] = predictions.argmax(axis=1)

from scipy.stats import ks_2samp
for svk in np.arange(num_class):
    ks_2samp(these_results.loc[unlabeled_indices][svk].values,these_results.loc[validation_set_indices][svk].values)


cv_res = lgb.cv(params, train_data, nfold=5, stratified=True, shuffle=True, verbose_eval=False, return_cvbooster=True)



list(cv_res['multi_logloss-mean'] == min(cv_res['multi_logloss-mean']))

list(compress(cv_res['multi_logloss-stdv'], list(cv_res['multi_logloss-mean'] == min(cv_res['multi_logloss-mean']))))


from scipy.stats import ks_2samp
pvals = np.array([kstest(data[i, :], 'uniform')[1] for i in range(100)]) # Use KS test to determine the p-value that they are drawn from a uniform distribution


for x in latent_distribution_columns:
    print(x)
    result = ks_2samp(latent_distribution_df.loc[unlabeled_indices][x].values,latent_distribution_df.loc[labeled_indices][x].values)
    result = pd.DataFrame([{'feature':x, 'statistic':result.statistic, 'pval':result.pvalue}])
    if x == latent_distribution_columns[0]:
        ks_results = result
    else:
        ks_results = pd.concat([ks_results, result])

keep_columns = ks_results.sort_values(by="pval", ascending=False)['feature'][:250].values

trials = Trials()
fmin_this_treatment = partial(score_impute_cv, X_train=X_train[keep_columns], y_train=y_train, all_features=list(keep_columns))
best = fmin(fmin_this_treatment,  space=space_lgb, algo=tpe.suggest, max_evals=args.hyperopt_max_evals, trials=trials, show_progressbar=True)

space_columns = list(space_lgb.keys())   
searched_space = pd.DataFrame(columns=['hyperopt_iteration'] + space_columns + ['loss'])
for idx, trial in enumerate(trials.trials):
    row = [idx]
    translated_eval = space_eval(space_lgb, {k: v[0] for k, v in trial['misc']['vals'].items()})
    for k in space_columns:
        row.append(translated_eval[k])
    row.append(trial['result']['loss'])
    searched_space.loc[idx] = row

searched_space['model_column'] = model_column
searched_space.to_csv(local_hyperopt_file, index=False)
# 
print('I am training the final model.')
params = space_eval(space_lgb, trials.argmin)
params.update(seed=47)
params['other_rate'] = params['other_rate'] * (1. - params['top_rate'])
train_data = lgb.Dataset(data=X_train[keep_columns], label=y_train, feature_name=list(keep_columns), categorical_feature=None, free_raw_data=True, params={'verbose':-1})
validation_data = lgb.Dataset(X_test[keep_columns], label=y_test, feature_name=list(keep_columns), categorical_feature=None, free_raw_data=True, params={'verbose':-1})
bst = lgb.train(params, train_data, valid_sets=[validation_data])



these_results = panel_pca_data[['uid','set_uid']]
these_results['set'] = ''
these_results.loc[these_results.uid.isin(list(Y_train['uid'].values.flatten())), "set"] = 'train'
these_results.loc[these_results.uid.isin(list(Y_test['uid'].values.flatten())), "set"] = 'test'
panel_pred_np = bst.predict(panel_libsvm_data,num_iteration=bst.best_iteration)
these_results["probPos"] = panel_pred_np





X_train = labeled_data_latent_distribution.loc[train_set_indices][latent_distribution_columns]






