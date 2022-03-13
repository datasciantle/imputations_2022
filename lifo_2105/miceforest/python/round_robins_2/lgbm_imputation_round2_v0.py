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
# # model = '381473'
# # model = '261625_261632'
# model = '381315'


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

parser.add_argument('--lgbm_max_iterations', default=200, type=int,
                    help='The maximum number of trees (recall that early stopping is also used).')

parser.add_argument('--hyperopt_max_evals', default=20, type=int,
                    help='The maximum number of evaluations for hyperparameter optimization in hyperopt.')

parser.add_argument('--delete_local', default=True, type=bool,
                    help='Whether or not to delete local output after pushing to s3, important to avoid blowing out limited memory of training instances.')

parser.add_argument('--verbose', default=False, type=bool,
                    help='Whether or not to print all output from training routines.  You probably should only use \'True\' for debugging.')

parser.add_argument('--redo', default=False, type=bool,
                    help='Whether or not to re-run a model that previously completed')

args = parser.parse_args()


#####################################################################################
###########                     focal loss                                ###########
#####################################################################################

class FocalLoss:
    
    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma
    
    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)
    
    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)
    
    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)
    
    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)
    
    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
    
        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1
        
        return (du * v + u * dv) * y * (pt * (1 - pt))
    
    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds
    
    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)
    
    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better




#####################################################################################
###########                      ETL BLOCK                                ###########
#####################################################################################
print('I am starting ETL and other preprocessing stages.')
# Here we collect model information.
s3 = s3fs.S3FileSystem()
model = args.model
multi = "_" in model
this_iteration = [x.split('_')[-1] for x in args.s3_feature_input_to_round.split('/') if 'iteration' in x][0]
was_completed = s3.isfile(args.base_s3_path + '/iteration_' + str(this_iteration) + '/focal_loss/completeness/' + args.model)

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
local_set_predictions_file = output_directory + '/' + model + '_predictions'
local_set_probabilities_file = output_directory + '/' + model + '_probabilities'
local_metrics_file = output_directory + '/' + model + '_metrics'
local_feature_importance_file = output_directory + '/' + model + '_featureImportance'
local_chi_square_file = output_directory + '/' + model + '_chiSquare'
local_distribution_file = output_directory + '/' + model + '_distributions'

exclude_attributes = ['129469','129406','129378', model]

feature_input_files = ['s3://' + x for x in s3.ls(args.s3_feature_input_to_round) if not '_SUCCESS' in x]
features = dd.read_csv(feature_input_files).compute()
features = features[[x for x in features.columns if x not in exclude_attributes]]

valid_respondents = pq.ParquetDataset(args.s3_valid_respondents, filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()[['uid','wavekey','respondentweight']]
valid_respondents['weight'] = valid_respondents['respondentweight'].values.astype(np.float64)

complete_taxonomy_df = pq.ParquetDataset(args.s3_complete_taxonomy_file, filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()
distributions_df = pq.ParquetDataset(args.s3_distributions, filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()
taxonomy = distributions_df.merge(complete_taxonomy_df[['sakey','svkey','attribute','value','proposed_round']], left_on=['sakey','svkey'], right_on=['sakey','svkey'])
taxonomy.loc[taxonomy.value.isnull(),'value'] = 2.
taxonomy['attribute_value'] = taxonomy['value'] - 1.
taxonomy['attribute_value'] = taxonomy['attribute_value'].astype('int')
taxonomy = taxonomy.drop(columns=['value'])


this_taxonomy = taxonomy.loc[taxonomy.attribute == model,]

for model_col in [model]:
    model_is_multi = '_' in model_col
    labels = pq.ParquetDataset(args.s3_attribute_matrix + '/attribute2=' + model_col, filesystem=s3, metadata_nthreads=10).read_pandas().to_pandas()[['uid','set_uid','label']]
    if not model_is_multi:
        labels.loc[(labels.label.notnull()),'label'] = labels.loc[(labels.label.notnull()),'label'] -1
    labels = labels[['uid','set_uid', 'label']]
    if model_col == model:
        imputation = labels
    else:
        imputation = imputation.merge(labels, left_on=['uid','set_uid'], right_on=['uid','set_uid'], how='outer')



X_all = imputation.merge(features, left_on=['uid','set_uid'], right_on=['uid','set_uid'], how='outer')
X_all = X_all.merge(valid_respondents[['uid','weight']], left_on=['uid'], right_on=['uid'])
categorical_columns = []
all_feature_columns = [x for x in X_all.columns if (not x in ['uid','set_uid','wavekey','weight','label'])]
for x in [x for x in X_all.columns if ((not x in ['uid','set_uid','wavekey','weight','label']) and ('ld_' not in x))]:
    print(x)
    X_all.loc[:,x] = X_all[x].astype('category') 
    categorical_columns.append(x)

num_class = np.unique(X_all.loc[X_all.label.notnull(),'label'].values).shape[0]
# Bootstrap distribution CIs
simulated_distributions =[]
for sim in np.arange(1000):
    data = X_all.loc[X_all.label.notnull()][['wavekey','label','weight']]
    sim = data.sample(n=data.shape[0], replace=True)
    this_sim = []
    for wave in np.unique(sim['wavekey']):
        wave_sim = sim.loc[sim.wavekey==wave]
        this_sim.append(np.bincount(wave_sim['label'], weights=wave_sim['weight'], minlength=num_class) / wave_sim['weight'].values.sum())
    simulated_distributions.append(np.array(this_sim).mean(axis=0))
    del data,sim

simulated_distributions = np.array(simulated_distributions)
dist_lower_bound = np.quantile(simulated_distributions, axis=0, q=0.025)
dist_upper_bound = np.quantile(simulated_distributions, axis=0, q=0.975)
simulated_distributions = simulated_distributions.mean(axis=0)

labeled_indices = X_all['label'].notnull().values
unlabeled_indices = X_all['label'].isnull().values
train_set_indices = ((X_all['set_uid']=='train') & (labeled_indices)).values
test_set_indices = ((X_all['set_uid']=='test') & (labeled_indices)).values
validation_set_indices = ((X_all['set_uid']=='validation') & (labeled_indices)).values
random_indices = X_all.sample(n=2000, replace=False)['uid'].values
random_indices = [(x in random_indices) for x in X_all.uid]

boolean_options = [True,False]
space_lgb = {
    # lightGBM parameters
    # Invariant parameter settings
    'zero_as_missing': False,
    'num_class': 1,
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

if not max([('ld_' in x) for x in all_feature_columns]):
    space_lgb.pop(max_bin)

space_features = dict()
for f in all_feature_columns:
     space_features.setdefault(f, hp.choice(f, boolean_options))

space_focal_loss = {
    'use_alpha': hp.choice('use_alpha', boolean_options),
    'alpha': hp.uniform('alpha', 0.0001, 1.0),
    'gamma': hp.uniform('gamma', 0.0001, 1.0)    
}

space_lgb['feature_params'] = space_features
space_lgb['focal_loss_params'] = space_focal_loss
space_svkeys = dict()
for svk in range(num_class):
     space_svkeys.setdefault(str(svk), hp.uniform(str(svk), 1e-10, 1.)**-1)

space_svkeys.setdefault('normalize', hp.choice('normalize', boolean_options))

def score_impute_focal_loss(impute_params, X_train, y_train, X_test, y_test, X_random, ignore_fail=False):
    print("Training GOSS LGBM with params: ")
    impute_params['other_rate'] = impute_params['other_rate'] * (1. - impute_params['top_rate'])
    feature_params  = impute_params.pop('feature_params')
    focal_loss_params = impute_params.pop('focal_loss_params')
    feature_columns = [x for x in feature_params.keys() if feature_params[x] is True]
    these_category_columns = [x for x in feature_columns if x in categorical_columns]
    print('I am using this many features: ' + str(len(feature_columns)))
    print(impute_params)
    alpha = focal_loss_params['alpha'] if focal_loss_params['use_alpha'] else None
    fl = FocalLoss(alpha=alpha,gamma=focal_loss_params['gamma'])
    train_data = lgb.Dataset(data=X_train[feature_columns], label=y_train, feature_name=feature_columns, categorical_feature=these_category_columns, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_train, fl.init_score(y_train), dtype=float))
    test_data = lgb.Dataset(data=X_test[feature_columns], label=y_test, feature_name=feature_columns, categorical_feature=these_category_columns, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_test, fl.init_score(y_train), dtype=float), reference=train_data)
    lgbm_model = lgb.train(params=impute_params, train_set=train_data, valid_sets=(train_data,test_data), valid_names=('train','test'),fobj=fl.lgb_obj, feval=fl.lgb_eval)
    y_pred = special.expit(fl.init_score(y_train) + lgbm_model.predict(X_test[feature_columns], num_iteration=lgbm_model.best_iteration))
    y_random = special.expit(fl.init_score(y_train) + lgbm_model.predict(X_random[feature_columns], num_iteration=lgbm_model.best_iteration))
    roc_score = metrics.roc_auc_score(y_test, y_pred)
    logloss_score = metrics.log_loss(y_test, y_pred)
    # We want to avoid models that only produce a few constant predictions, so let's see how many unique probabilities are in a batch
    random_score = 1. - (len(np.unique(y_random))/len(y_random))
    print(f"Test's ROC AUC: {roc_score:.5f}")
    print(f"Test's logloss: {logloss_score:.5f}")
    if ((roc_score == 0.5) & (ignore_fail is False)):
        print('This trial was a FAILURE.')
        return {'loss': logloss_score + random_score, 'logloss':logloss_score, 'roc_score':roc_score, 'random_score':random_score, 'status':STATUS_FAIL}
    else:
        return {'loss': logloss_score + random_score, 'logloss':logloss_score, 'roc_score':roc_score, 'random_score':random_score,  'status':STATUS_OK}


best_params_dict = dict()
metrics_dict = dict()
if multi:
    gc.collect()
    y_train_binary = X_all.loc[train_set_indices,'label'].values
    y_test_binary = X_all.loc[test_set_indices,'label'].values
    X_train = X_all.loc[train_set_indices]
    X_test = X_all.loc[test_set_indices]
    X_random = X_all.loc[random_indices]
    try:
        trials = Trials()
        fmin_this_treatment = partial(score_impute_focal_loss, X_train=X_train, y_train=y_train_binary, X_test=X_test, y_test=y_test_binary, X_random=X_random, ignore_fail=False)
        best = fmin(fmin_this_treatment,  space=space_lgb, algo=tpe.suggest, max_evals=args.hyperopt_max_evals , trials=trials, show_progressbar=True)
        hyperopt_success = False
    except:
        trials = Trials()
        fmin_this_treatment = partial(score_impute_focal_loss, X_train=X_train, y_train=y_train_binary, X_test=X_test, y_test=y_test_binary, X_random=X_random, ignore_fail=True)
        best = fmin(fmin_this_treatment,  space=space_lgb, algo=tpe.suggest, max_evals=args.hyperopt_max_evals , trials=trials, show_progressbar=True)
        hyperopt_success = True
    
    del X_train, X_test
    gc.collect()
    best_params = space_eval(space_lgb, trials.argmin)
    best_params_dict[str(1)] = best_params
    metrics_dict[str(1)] = trials.best_trial['result']

if not multi:
    for i in np.arange(num_class):
        gc.collect()
        y_train_binary = LabelBinarizer(sparse_output=False).fit_transform(X_all.loc[train_set_indices,'label'].values)[:,i]
        y_test_binary = LabelBinarizer(sparse_output=False).fit_transform(X_all.loc[test_set_indices,'label'].values)[:,i]
        X_train = X_all.loc[train_set_indices]
        X_test = X_all.loc[test_set_indices]
        X_random = X_all.loc[random_indices]
        hyperopt_success = []
        try:
            trials = Trials()
            fmin_this_treatment = partial(score_impute_focal_loss, X_train=X_train, y_train=y_train_binary, X_test=X_test, y_test=y_test_binary, X_random=X_random, ignore_fail=False)
            best = fmin(fmin_this_treatment,  space=space_lgb, algo=tpe.suggest, max_evals=args.hyperopt_max_evals , trials=trials, show_progressbar=True)
            hyperopt_success.append(False)
        except:
            trials = Trials()
            fmin_this_treatment = partial(score_impute_focal_loss, X_train=X_train, y_train=y_train_binary, X_test=X_test, y_test=y_test_binary, X_random=X_random, ignore_fail=True)
            best = fmin(fmin_this_treatment,  space=space_lgb, algo=tpe.suggest, max_evals=args.hyperopt_max_evals , trials=trials, show_progressbar=True)
            hyperopt_success.append(False)
        
        del X_train, X_test
        gc.collect()
        hyperopt_success = min(hyperopt_success)
        best_params = space_eval(space_lgb, trials.argmin)
        best_params_dict[str(i)] = best_params
        metrics_dict[str(i)] = trials.best_trial['result']


predictions_dict = dict()
if  multi:
    gc.collect()
    y_train_binary = X_all.loc[train_set_indices,'label'].values
    y_test_binary = X_all.loc[test_set_indices,'label'].values
    X_train = X_all.loc[train_set_indices]
    X_test = X_all.loc[test_set_indices]
    # X_train = X_all.loc[train_set_indices][feature_columns]
    # X_test = X_all.loc[test_set_indices][feature_columns]
    params = best_params_dict[str(1)].copy()
    feature_params  = params.pop('feature_params')
    focal_loss_params = params.pop('focal_loss_params')
    feature_columns = [x for x in feature_params.keys() if feature_params[x] is True]
    params.update(seed=47)
    params['other_rate'] = params['other_rate'] * (1. - params['top_rate'])
    print(params)
    print(focal_loss_params)
    alpha = focal_loss_params['alpha'] if focal_loss_params['use_alpha'] else None
    fl = FocalLoss(alpha=alpha,gamma=focal_loss_params['gamma'])
    train_data = lgb.Dataset(data=X_train[feature_columns], label=y_train_binary, feature_name=feature_columns, categorical_feature=feature_columns, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_train_binary, fl.init_score(y_train_binary), dtype=float))
    test_data = lgb.Dataset(data=X_test[feature_columns], label=y_test_binary, feature_name=feature_columns, categorical_feature=feature_columns, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_test_binary, fl.init_score(y_train_binary), dtype=float), reference=train_data)
    lgbm_model = lgb.train(params=params, train_set=train_data, valid_sets=(train_data,test_data), valid_names=('train','test'),fobj=fl.lgb_obj, feval=fl.lgb_eval)
    del X_train, X_test, train_data, test_data
    gc.collect()
    predictions_dict[str(1)] = special.expit(fl.init_score(y_train_binary) + lgbm_model.predict(X_all[feature_columns], num_iteration=lgbm_model.best_iteration))
    feature_importance_split = pd.DataFrame({'feature':feature_columns, 'importance': lgbm_model.feature_importance(importance_type='split', iteration=lgbm_model.best_iteration), 'importance_type':'split'})
    feature_importance_gain = pd.DataFrame({'feature':feature_columns, 'importance': lgbm_model.feature_importance(importance_type='gain', iteration=lgbm_model.best_iteration), 'importance_type':'gain'})
    feature_importance = feature_importance_split.append(feature_importance_gain)
    feature_importance['model'] = model
    feature_importance['svkey_index'] = 1
    feature_importance['best_iteration'] = lgbm_model.best_iteration
    feature_importance_df = feature_importance
    predictions = pd.DataFrame(predictions_dict)
    predictions['0'] = 1. - predictions['1'].values
    these_results = X_all[['uid','set_uid','label','wavekey','weight']]
    these_results['attribute_model'] = model
    these_results = pd.concat([these_results, predictions], axis=1)
    fpr, tpr, thresholds = metrics.roc_curve(y_test_binary, predictions[test_set_indices]['1'].values)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    mcc = []
    for th in thresholds:
        y_p = np.array([1. if x >= th else 0. for x in predictions[test_set_indices]['1'].values])
        mcc.append(metrics.matthews_corrcoef(y_test_binary,y_p))
    
    optimal_idx_mcc = np.argmax(mcc)
    optimal_threshold_mcc = thresholds[optimal_idx_mcc]
    these_results['max_prediction'] = [1. if x >= optimal_threshold_mcc else 0. for x in predictions['1'].values]
    final_metrics_dict = dict()
    for s in ['train','test','validation']:
        res = these_results.loc[((these_results['set_uid']==s) & (labeled_indices))]
        roc = metrics.roc_auc_score(res['label'].values,res[str(1)])
        ll = metrics.log_loss(res['label'].values,res[str(1)])
        final_metrics_dict[str(1) + '_' + s] = {'logloss':ll,'roc_score':roc}
    
    fmdf = pd.DataFrame(final_metrics_dict).transpose()
    fmdf['data_slice'] = fmdf.index
    hmdf = pd.DataFrame(metrics_dict).transpose()[['logloss','roc_score']]
    hmdf['data_slice'] = [x + '_hyperopt' for x in hmdf.index]
    all_these_metrics = pd.concat([hmdf,fmdf])
    all_these_metrics['model'] = model

if not multi:
    for i in np.arange(num_class):
        gc.collect()
        y_train_binary = LabelBinarizer(sparse_output=False).fit_transform(X_all.loc[train_set_indices,'label'].values)[:,i]
        y_test_binary = LabelBinarizer(sparse_output=False).fit_transform(X_all.loc[test_set_indices,'label'].values)[:,i]
        X_train = X_all.loc[train_set_indices]
        X_test = X_all.loc[test_set_indices]
        params = best_params_dict[str(i)].copy()
        feature_params  = params.pop('feature_params')
        focal_loss_params = params.pop('focal_loss_params')
        feature_columns = [x for x in feature_params.keys() if feature_params[x] is True]
        params.update(seed=47)
        params['other_rate'] = params['other_rate'] * (1. - params['top_rate'])
        print(params)
        print(focal_loss_params)
        alpha = focal_loss_params['alpha'] if focal_loss_params['use_alpha'] else None
        fl = FocalLoss(alpha=alpha,gamma=focal_loss_params['gamma'])
        train_data = lgb.Dataset(data=X_train[feature_columns], label=y_train_binary, feature_name=feature_columns, categorical_feature=feature_columns, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_train_binary, fl.init_score(y_train_binary), dtype=float))
        test_data = lgb.Dataset(data=X_test[feature_columns], label=y_test_binary, feature_name=feature_columns, categorical_feature=feature_columns, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_test_binary, fl.init_score(y_train_binary), dtype=float), reference=train_data)
        lgbm_model = lgb.train(params=params, train_set=train_data, valid_sets=(train_data,test_data), valid_names=('train','test'),fobj=fl.lgb_obj, feval=fl.lgb_eval)
        predictions_dict[str(i)] = special.expit(fl.init_score(y_train_binary) + lgbm_model.predict(X_all[feature_columns], num_iteration=lgbm_model.best_iteration))
        feature_importance_split = pd.DataFrame({'feature':feature_columns, 'importance': lgbm_model.feature_importance(importance_type='split', iteration=lgbm_model.best_iteration), 'importance_type':'split'})
        feature_importance_gain = pd.DataFrame({'feature':feature_columns, 'importance': lgbm_model.feature_importance(importance_type='gain', iteration=lgbm_model.best_iteration), 'importance_type':'gain'})
        feature_importance = feature_importance_split.append(feature_importance_gain)
        feature_importance['model'] = model
        feature_importance['svkey_index'] = i
        feature_importance['best_iteration'] = lgbm_model.best_iteration
        if i == 0:
            feature_importance_df = feature_importance
        else:
            feature_importance_df = pd.concat([feature_importance_df,feature_importance])
    
    predictions = pd.DataFrame(predictions_dict)
    these_results = X_all[['uid','set_uid','label','wavekey','weight']]
    these_results['attribute_model'] = model
    these_results = pd.concat([these_results, predictions], axis=1)
    these_results['max_prediction'] = predictions.values.argmax(axis=1)
    final_metrics_dict = dict()
    for s in ['train','test','validation']:
        res = these_results.loc[((these_results['set_uid']==s) & (labeled_indices))]
        for i in np.arange(num_class):
            roc = metrics.roc_auc_score(LabelBinarizer(sparse_output=False).fit_transform(res['label'].values)[:,i],res[str(i)])
            ll = metrics.log_loss(LabelBinarizer(sparse_output=False).fit_transform(res['label'].values)[:,i],res[str(i)])
            final_metrics_dict[str(i) + '_' + s] = {'logloss':ll,'roc_score':roc}
    
    fmdf = pd.DataFrame(final_metrics_dict).transpose()
    fmdf['data_slice'] = fmdf.index
    hmdf = pd.DataFrame(metrics_dict).transpose()[['logloss','roc_score']]
    hmdf['data_slice'] = [x + '_hyperopt' for x in hmdf.index]
    all_these_metrics = pd.concat([hmdf,fmdf])
    all_these_metrics['model'] = model


def set_imputation_thresholds(svkey_params, result_df, num_class, dist_lower_bound, dist_upper_bound, simulated_distributions):
    print("Tuning imputation predictions...")
    labeled = np.array([0. if x is True else 1. for x in result_df['label'].notnull().values])
    pred_np = result_df[[str(x) for x in range(num_class)]].values
    scaled_predictions = pred_np / pred_np.max(1,keepdims=True)
    if svkey_params['normalize']:
        pred_np = pred_np / np.linalg.norm(pred_np, ord=2, axis=1, keepdims=True)
    vec = np.array([svkey_params[str(x)] for x in range(num_class)])
    cutoffs = vec * pred_np
    winners = np.argmax(cutoffs, axis=1)
    result_df['imputation'] = winners
    result_df.loc[result_df.label.notnull(), 'imputation'] = result_df.loc[result_df.label.notnull(), 'label']
    prob_loss = 1. - np.take_along_axis(scaled_predictions, winners.reshape(-1, 1), 1).flatten()
    prob_loss = prob_loss * labeled
    prob_loss = prob_loss.sum() / len(prob_loss)
    current_distributions = np.bincount(result_df['imputation'].values.astype(int), weights=result_df['weight'].values, minlength=num_class) / result_df['weight'].values.sum()
    distribution_error = sum((simulated_distributions-current_distributions)**2)
    confidence_interval_error = (current_distributions >= dist_lower_bound) & (current_distributions <= dist_upper_bound).astype(int)
    confidence_interval_error = 1. - confidence_interval_error.mean()
    imputation_error = distribution_error + prob_loss + confidence_interval_error
    print(f"Imputation error is: {imputation_error:.5f}")
    return {'loss': imputation_error, 'distribution_error':distribution_error, 'prob_loss':prob_loss, 'confidence_interval_error':confidence_interval_error, 'status': STATUS_OK}


waves = np.unique(these_results['wavekey'])
wave_distribution_dict = dict()
for wave in waves:
    this_wave = these_results.loc[these_results.wavekey==wave]
    this_wave['selected_value'] = this_wave['label']
    if max(this_wave.label.isnull()):
        print('Wave ' + str(wave) + ' contains missing data.  Setting imputations now.')
        wave_trials = Trials()
        fmin_this_wave = partial(set_imputation_thresholds, result_df=this_wave, num_class=num_class, dist_lower_bound=dist_lower_bound, dist_upper_bound=dist_upper_bound, simulated_distributions=simulated_distributions)
        wave_best = fmin(fmin_this_wave,  space=space_svkeys, algo=tpe.suggest, max_evals=args.hyperopt_max_evals*num_class , trials=wave_trials, show_progressbar=True)
        wave_params = space_eval(space_svkeys, wave_trials.argmin)
        vec = np.array([wave_params[str(x)] for x in range(num_class)])
        pred_np = this_wave[[str(x) for x in range(num_class)]].values
        cutoffs = vec * pred_np
        winners = np.argmax(cutoffs, axis=1)
        this_wave['selected_value'] = winners
        this_wave.loc[this_wave.label.notnull(), 'selected_value'] = this_wave.loc[this_wave.label.notnull(), 'label']
        this_wave.loc[this_wave.imputation.notnull(), 'selected_value'] = this_wave.loc[this_wave.imputation.notnull(), 'selected_value'].values.astype(int)
    if wave == waves[0]:
        imputed_results = this_wave[['uid','selected_value']]
    else:
        imputed_results = pd.concat([imputed_results,this_wave[['uid','selected_value']]])


these_results = these_results.merge(imputed_results, right_on=['uid'], left_on=['uid'])
wave_distribution_dict = dict()
for wave in waves:
    this_wave = these_results.loc[these_results.wavekey==wave]
    wave_distribution_dict[str(wave)] = np.bincount(this_wave.selected_value, weights=this_wave.weight, minlength=num_class) / this_wave.weight.sum()

wave_distribution_df = pd.DataFrame(wave_distribution_dict)
wave_distribution_df['imputed_distribution'] = wave_distribution_df.values.mean(axis=1)
wave_distribution_df['simulation_distribution'] = simulated_distributions
wave_distribution_df['lower_bound_distribution'] = dist_lower_bound
wave_distribution_df['upper_bound_distribution'] = dist_upper_bound
wave_distribution_df['attribute'] = model
wave_distribution_df['attribute_value'] = wave_distribution_df.index
wave_distribution_df = taxonomy.merge(wave_distribution_df, left_on=['attribute','attribute_value'], right_on=['attribute','attribute_value'])

print(wave_distribution_df)
print(all_these_metrics)

initial_feature_selection = feature_selection.chi2(X_all.loc[labeled_indices][categorical_columns], X_all.loc[labeled_indices,'label'])
post_feature_selection = feature_selection.chi2(X_all[categorical_columns], these_results['selected_value'])
ifs_df = pd.concat([pd.DataFrame(categorical_columns, columns=['attribute']),pd.DataFrame(initial_feature_selection[0], columns=['chi2']), pd.DataFrame(initial_feature_selection[1], columns=['p_value'])],axis=1)
ifs_df['attribute_model'] = args.model
pfs_df = pd.concat([pd.DataFrame(categorical_columns, columns=['attribute']),pd.DataFrame(post_feature_selection[0], columns=['chi2']), pd.DataFrame(post_feature_selection[1], columns=['p_value'])],axis=1)
pfs_df['attribute_model'] = args.model
ifs_df.to_csv(local_chi_square_file+'_preImputation', index=False)
pfs_df.to_csv(local_chi_square_file+'_postImputation', index=False)


# Time to save everything
these_results.to_csv(local_set_predictions_file, columns=['uid','set_uid','wavekey','attribute_model','label','max_prediction','selected_value'], index=False)
these_results.to_csv(local_set_probabilities_file, columns=['uid','set_uid','wavekey','attribute_model','label'] + [[str(1)] if multi else [str(x) for x in range(num_class)]][0], index=False)
all_these_metrics.to_csv(local_metrics_file, index=False)
feature_importance.to_csv(local_feature_importance_file, index=False)
wave_distribution_df.to_csv(local_distribution_file, index=False)

final_sync_cmd = 'aws s3 sync ' + output_directory + ' ' + args.base_s3_path + '/iteration_' + str(this_iteration) + '/focal_loss/results/' + args.model
os.system(final_sync_cmd)

finish_time = time.time()
times_df = pd.DataFrame([{'model':args.model,'num_class':num_class, 'iteration':this_iteration, 'start_time':start_time, 'finish_time':finish_time, 'total_time':finish_time-start_time, 'hyperopt_success':hyperopt_success}])
times_df.to_csv(args.base_s3_path + '/iteration_' + str(this_iteration) + '/focal_loss/completeness/' + args.model, index=False)

if args.delete_local:
    print('Cleaning up local')
    shutil.rmtree(output_directory)

quit()