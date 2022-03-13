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
from sklearn import metrics

# Machine learning modules
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe, pyll, space_eval
from hyperopt.pyll import scope
import xgboost as xgb
import lightgbm as lgb

## Defaults for prototyping in shell
# base_s3_path = 's3://datasci-scantle/unified_ncs/experiments/2105'
# model_version = 'lifo_2105'
# s3_pickle_file = 's3://datasci-scantle/unified_ncs/experiments/2105/python_inputs/iteration_0.pickle'
# model = '310152_310154'

base_s3_path = 'fake'
model_version = 'fake'
s3_pickle_file = 'fake'
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
parser.add_argument('--s3_pickle_file', default=s3_pickle_file, type=str,
                    help='The s3 location of the python dataset environment file.')
## model parameters
parser.add_argument('--model', default=model, type=str,
                    help='The particular model you are training, written as sakey_svkey_behaviorType.')

parser.add_argument('--lgbm_max_iterations', default=200, type=int,
                    help='The maximum number of trees (recall that early stopping is also used).')

parser.add_argument('--hyperopt_max_evals', default=20, type=int,
                    help='The maximum number of evaluations for hyperparameter optimization in hyperopt.')

parser.add_argument('--delete_local', default=True, type=bool,
                    help='Whether or not to delete local output after pushing to s3, important to avoid blowing out limited memory of training instances.')

parser.add_argument('--verbose', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='Whether or not to print all output from training routines.  You probably should only use \'True\' for debugging.')

args = parser.parse_args()




#####################################################################################
###########                     focal loss                                ###########
#####################################################################################
from scipy import optimize
from scipy import special
from joblib import Parallel, delayed
from sklearn.multiclass import _ConstantPredictor
from sklearn.preprocessing import LabelBinarizer
from scipy import special

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
model = args.model
multi = "_" in model

working_directory = os.getenv('HOME') + '/' + args.model_version
output_directory = working_directory + '/output/' + args.model
os.makedirs(working_directory, exist_ok=True)
os.makedirs(working_directory + '/inputs', exist_ok=True)
os.makedirs(output_directory, exist_ok=True)


# Here we enumerate the output files
local_pickle_file = working_directory + '/inputs/local_pickle_data.pickle'
local_set_predictions_file = output_directory + '/' + model + '_predictions'
local_set_probabilities_file = output_directory + '/' + model + '_probabilities'
local_metrics_file = output_directory + '/' + model + '_metrics'
local_feature_importance_file = output_directory + '/' + model + '_featureImportance'
local_chi_square_file = output_directory + '/' + model + '_chiSquare'


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
mostly_complete_level_0 = model_plan_df.loc[((model_plan_df['n_respondents'] > X.shape[0] * 0.6) & (model_plan_df['proposed_round']==0))]
include_these_attributes = complete_taxonomy.loc[((complete_taxonomy.svkey.isin(mostly_complete_level_0['svkey'].values)) & (~complete_taxonomy.attribute.isin(exclude_attributes)))][['attribute','final_index']].drop_duplicates().sort_values(by='final_index')
raw_attributes = list(include_these_attributes['attribute'].values)

label_index = complete_taxonomy.loc[(complete_taxonomy['attribute']==args.model)]['final_index'].values[0]
y_initial = np.array(X[:,label_index].todense()).flatten()
labeled_indices = y_initial != 0
unlabeled_indices = y_initial == 0
latent_distribution_df['is_labeled'] = 0
latent_distribution_df.loc[labeled_indices,'is_labeled'] = 1
latent_distribution_df['label'] = y_initial
latent_distribution_df.loc[unlabeled_indices,'label'] = np.nan
latent_distribution_df.loc[((labeled_indices)),'label'] = latent_distribution_df.loc[((labeled_indices)),'label'] - 1.

latent_distribution_df = pd.concat([latent_distribution_df, pd.DataFrame(X[:,include_these_attributes['final_index'].values].todense(), columns=raw_attributes).replace(0.0, np.nan)], axis=1)

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


labeled_data = X[labeled_indices,]
initial_feature_selection = feature_selection.chi2(labeled_data, np.array(X[:,label_index].todense()).flatten()[labeled_indices]-1)
ifs_df = pd.concat([pd.DataFrame(list(np.arange(0,X.shape[1])), columns=['feature_index']),pd.DataFrame(initial_feature_selection[0], columns=['chi2']), pd.DataFrame(initial_feature_selection[1], columns=['p_value'])],axis=1)
ifs_df = ifs_df.merge(complete_taxonomy, left_on=['feature_index'], right_on=['final_index'])[['sakey','svkey','multi','final_index','chi2','p_value','is_display_logic_sakey', 'is_display_logic_svkey',  'min_level',  'proposed_round']]
ifs_df['attribute'] = args.model
ifs_df.to_csv(local_chi_square_file, index=False)
del labeled_data
gc.collect()

num_class = len(np.unique(latent_distribution_df[labeled_indices]['label'].values))
objective_function = 'binary' if num_class < 3 else 'multiclassova'
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

space_features = dict()
for f in latent_distribution_columns:
     space_features.setdefault(f, hp.choice(f, boolean_options))

space_focal_loss = {
    'use_alpha': hp.choice('use_alpha', boolean_options),
    'alpha': hp.uniform('alpha', 0.0001, 1.0),
    'gamma': hp.uniform('gamma', 0.0001, 1.0)    
}

space_lgb['latent_distribution_columns'] = space_features
space_lgb['focal_loss_params'] = space_focal_loss

def score_impute_focal_loss(impute_params, X_train, y_train, X_test, y_test):
    print("Training GOSS LGBM with params: ")
    ld_columns = impute_params.pop('latent_distribution_columns')
    keep_features = [x for x in latent_distribution_columns if ld_columns[x] is True] + raw_attributes
    impute_params['other_rate'] = impute_params['other_rate'] * (1. - impute_params['top_rate'])
    focal_loss_params = impute_params.pop('focal_loss_params')
    print(impute_params)
    alpha = focal_loss_params['alpha'] if focal_loss_params['use_alpha'] else None
    fl = FocalLoss(alpha=alpha,gamma=focal_loss_params['gamma'])
    train_data = lgb.Dataset(data=X_train[keep_features], label=y_train, feature_name=keep_features, categorical_feature=raw_attributes, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_train, fl.init_score(y_train), dtype=float))
    test_data = lgb.Dataset(data=X_test[keep_features], label=y_test, feature_name=keep_features, categorical_feature=raw_attributes, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_test, fl.init_score(y_train), dtype=float), reference=train_data)
    lgbm_model = lgb.train(params=impute_params, train_set=train_data, valid_sets=(train_data,test_data), valid_names=('train','test'),fobj=fl.lgb_obj, feval=fl.lgb_eval)
    y_pred = special.expit(fl.init_score(y_train) + lgbm_model.predict(X_test[keep_features], num_iteration=lgbm_model.best_iteration))
    roc_score = metrics.roc_auc_score(y_test, y_pred)
    logloss_score = metrics.log_loss(y_test, y_pred)
    print(f"Test's ROC AUC: {roc_score:.5f}")
    print(f"Test's logloss: {logloss_score:.5f}")
    if roc_score == 0.5:
        return {'loss': logloss_score, 'roc_score':roc_score, 'status': STATUS_FAIL}
    else:
        return {'loss': logloss_score, 'roc_score':roc_score, 'status': STATUS_OK}


best_params_dict = dict()
metrics_dict = dict()
if multi:
    y_train_binary = y_train
    y_test_binary = y_test
    trials = Trials()
    fmin_this_treatment = partial(score_impute_focal_loss, X_train=X_train, y_train=y_train_binary, X_test=X_test, y_test=y_test_binary)
    best = fmin(fmin_this_treatment,  space=space_lgb, algo=tpe.suggest, max_evals=args.hyperopt_max_evals , trials=trials, show_progressbar=True)
    best_params = space_eval(space_lgb, trials.argmin)
    best_params_dict[str(1)] = best_params
    metrics_dict[str(1)] = trials.best_trial['result']

if not multi:
    for i in np.arange(num_class):
        y_train_binary = LabelBinarizer(sparse_output=False).fit_transform(y_train)[:,i]
        y_test_binary = LabelBinarizer(sparse_output=False).fit_transform(y_test)[:,i]
        trials = Trials()
        fmin_this_treatment = partial(score_impute_focal_loss, X_train=X_train, y_train=y_train_binary, X_test=X_test, y_test=y_test_binary)
        best = fmin(fmin_this_treatment,  space=space_lgb, algo=tpe.suggest, max_evals=args.hyperopt_max_evals , trials=trials, show_progressbar=True)
        best_params = space_eval(space_lgb, trials.argmin)
        best_params_dict[str(i)] = best_params
        metrics_dict[str(i)] = trials.best_trial['result']

predictions_dict = dict()
if  multi:
    y_train_binary = y_train
    y_test_binary = y_test
    params = best_params_dict[str(1)].copy()
    ld_columns = params.pop('latent_distribution_columns')
    focal_loss_params = params.pop('focal_loss_params')
    params.update(seed=47)
    keep_features = [x for x in latent_distribution_columns if ld_columns[x] is True] + raw_attributes
    params['other_rate'] = params['other_rate'] * (1. - params['top_rate'])
    print(params)
    print(focal_loss_params)
    alpha = focal_loss_params['alpha'] if focal_loss_params['use_alpha'] else None
    fl = FocalLoss(alpha=alpha,gamma=focal_loss_params['gamma'])
    train_data = lgb.Dataset(data=X_train[keep_features], label=y_train_binary, feature_name=keep_features, categorical_feature=raw_attributes, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_train_binary, fl.init_score(y_train_binary), dtype=float))
    test_data = lgb.Dataset(data=X_test[keep_features], label=y_test_binary, feature_name=keep_features, categorical_feature=raw_attributes, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_test_binary, fl.init_score(y_train_binary), dtype=float), reference=train_data)
    lgbm_model = lgb.train(params=params, train_set=train_data, valid_sets=(train_data,test_data), valid_names=('train','test'),fobj=fl.lgb_obj, feval=fl.lgb_eval)
    predictions_dict[str(1)] = special.expit(fl.init_score(y_train_binary) + lgbm_model.predict(latent_distribution_df[keep_features], num_iteration=lgbm_model.best_iteration))
    feature_importance_split = pd.DataFrame({'feature':keep_features, 'importance': lgbm_model.feature_importance(importance_type='split', iteration=lgbm_model.best_iteration), 'importance_type':'split'})
    feature_importance_gain = pd.DataFrame({'feature':keep_features, 'importance': lgbm_model.feature_importance(importance_type='gain', iteration=lgbm_model.best_iteration), 'importance_type':'gain'})
    feature_importance = feature_importance_split.append(feature_importance_gain)
    feature_importance['model'] = model
    feature_importance['svkey_index'] = 1
    feature_importance['best_iteration'] = lgbm_model.best_iteration
    feature_importance_df = feature_importance
    predictions = pd.DataFrame(predictions_dict)
    these_results = latent_distribution_df[['uid','set_uid','is_labeled','label']]
    these_results['attribute_model'] = model
    these_results = pd.concat([these_results, predictions], axis=1)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions[test_set_indices].values)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    mcc = []
    for th in thresholds:
        y_p = np.array([1. if x >= th else 0. for x in predictions[test_set_indices].values])
        mcc.append(metrics.matthews_corrcoef(y_test,y_p))
    
    optimal_idx_mcc = np.argmax(mcc)
    optimal_threshold_mcc = thresholds[optimal_idx_mcc]
    these_results['max_prediction'] = [1. if x >= optimal_threshold_mcc else 0. for x in predictions.values]
    these_results['selected_value'] = these_results['max_prediction']
    these_results.loc[labeled_indices,'selected_value'] = these_results.loc[labeled_indices]['max_prediction']
    final_metrics_dict = dict()
    for s in ['train','test','validation']:
        res = these_results.loc[((these_results['set_uid']==s) & (these_results['is_labeled']==1))]
        roc = metrics.roc_auc_score(res['label'].values,res[str(1)])
        ll = metrics.log_loss(res['label'].values,res[str(1)])
        final_metrics_dict[str(1) + '_' + s] = {'loss':ll,'roc_score':roc}
    
    fmdf = pd.DataFrame(final_metrics_dict).transpose()
    fmdf['data_slice'] = fmdf.index
    hmdf = pd.DataFrame(metrics_dict).transpose()[['loss','roc_score']]
    hmdf['data_slice'] = [x + '_hyperopt' for x in hmdf.index]
    all_these_metrics = pd.concat([hmdf,fmdf])
    all_these_metrics['model'] = model

if not multi:
    for i in np.arange(num_class):
        y_train_binary = LabelBinarizer(sparse_output=False).fit_transform(y_train)[:,i]
        y_test_binary = LabelBinarizer(sparse_output=False).fit_transform(y_test)[:,i]
        params = best_params_dict[str(i)].copy()
        ld_columns = params.pop('latent_distribution_columns')
        focal_loss_params = params.pop('focal_loss_params')
        params.update(seed=47)
        keep_features = [x for x in latent_distribution_columns if ld_columns[x] is True] + raw_attributes
        params['other_rate'] = params['other_rate'] * (1. - params['top_rate'])
        print(params)
        print(focal_loss_params)
        alpha = focal_loss_params['alpha'] if focal_loss_params['use_alpha'] else None
        fl = FocalLoss(alpha=alpha,gamma=focal_loss_params['gamma'])
        train_data = lgb.Dataset(data=X_train[keep_features], label=y_train_binary, feature_name=keep_features, categorical_feature=raw_attributes, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_train_binary, fl.init_score(y_train_binary), dtype=float))
        test_data = lgb.Dataset(data=X_test[keep_features], label=y_test_binary, feature_name=keep_features, categorical_feature=raw_attributes, free_raw_data=True, params={'verbose':-1}, init_score=np.full_like(y_test_binary, fl.init_score(y_train_binary), dtype=float), reference=train_data)
        lgbm_model = lgb.train(params=params, train_set=train_data, valid_sets=(train_data,test_data), valid_names=('train','test'),fobj=fl.lgb_obj, feval=fl.lgb_eval)
        predictions_dict[str(i)] = special.expit(fl.init_score(y_train_binary) + lgbm_model.predict(latent_distribution_df[keep_features], num_iteration=lgbm_model.best_iteration))
        feature_importance_split = pd.DataFrame({'feature':keep_features, 'importance': lgbm_model.feature_importance(importance_type='split', iteration=lgbm_model.best_iteration), 'importance_type':'split'})
        feature_importance_gain = pd.DataFrame({'feature':keep_features, 'importance': lgbm_model.feature_importance(importance_type='gain', iteration=lgbm_model.best_iteration), 'importance_type':'gain'})
        feature_importance = feature_importance_split.append(feature_importance_gain)
        feature_importance['model'] = model
        feature_importance['svkey_index'] = i
        feature_importance['best_iteration'] = lgbm_model.best_iteration
        if i == 0:
            feature_importance_df = feature_importance
        else:
            feature_importance_df = pd.concat([feature_importance_df,feature_importance])
    
    predictions = pd.DataFrame(predictions_dict)
    these_results = latent_distribution_df[['uid','set_uid','is_labeled','label']]
    these_results['attribute_model'] = model
    these_results = pd.concat([these_results, predictions], axis=1)
    these_results['max_prediction'] = predictions.values.argmax(axis=1)
    these_results['selected_value'] = these_results['max_prediction']
    these_results.loc[labeled_indices,'selected_value'] = these_results.loc[labeled_indices]['max_prediction']
    final_metrics_dict = dict()
    for s in ['train','test','validation']:
        res = these_results.loc[((these_results['set_uid']==s) & (these_results['is_labeled']==1))]
        for i in np.arange(num_class):
            roc = metrics.roc_auc_score(LabelBinarizer(sparse_output=False).fit_transform(res['label'].values)[:,i],res[str(i)])
            ll = metrics.log_loss(LabelBinarizer(sparse_output=False).fit_transform(res['label'].values)[:,i],res[str(i)])
            final_metrics_dict[str(i) + '_' + s] = {'loss':ll,'roc_score':roc}
    
    fmdf = pd.DataFrame(final_metrics_dict).transpose()
    fmdf['data_slice'] = fmdf.index
    hmdf = pd.DataFrame(metrics_dict).transpose()[['loss','roc_score']]
    hmdf['data_slice'] = [x + '_hyperopt' for x in hmdf.index]
    all_these_metrics = pd.concat([hmdf,fmdf])
    all_these_metrics['model'] = model

# Time to save everything
these_results.to_csv(local_set_predictions_file, columns=['uid','set_uid','attribute_model','is_labeled','label','max_prediction','selected_value'], index=False)
these_results.to_csv(local_set_probabilities_file, columns=['uid','attribute_model','is_labeled','label'] + [str(1)], index=False)
all_these_metrics.to_csv(local_metrics_file, index=False)
feature_importance.to_csv(local_feature_importance_file, index=False)

final_sync_cmd = 'aws s3 sync ' + output_directory + ' ' + args.base_s3_path + '/iteration_' + str(args.iteration) + '/' + 'results/' + args.model
os.system(final_sync_cmd)

finish_time = time.time()
times_df = pd.DataFrame([{'model':args.model,'num_class':num_class, 'iteration':args.iteration, 'start_time':start_time, 'finish_time':finish_time, 'total_time':finish_time-start_time}])
times_df.to_csv(args.base_s3_path + '/iteration_' + str(args.iteration) + '/' + 'completeness/' + args.model, index=False)

if args.delete_local:
    print('Cleaning up local')
    shutil.rmtree(output_directory)

