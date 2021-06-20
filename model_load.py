# %%
import os.path

import eli5
import joblib
import numpy
import pandas
from eli5.sklearn import PermutationImportance
from google_drive_downloader import GoogleDriveDownloader as gdd
from IPython import get_ipython
from mlxtend.feature_selection import ColumnSelector
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sqlalchemy import select
from sqlalchemy.sql.expression import true

from source.custom_transfomers.debug_transformer import Debug
from source.custom_transfomers.time_window_transformer import \
    TimeWindowTransformer
from source.data_base.connection import session
from source.data_base.models import models
from source.project_utils.constants import targets, targets_models
from source.project_utils.data_manipulation import generate_aggregation

pandas.set_option('display.max_columns', 51)

# %%
# Carregar o modelo
regression_models = {}
for target, _ in targets_models.items():
    regression_models[target] = joblib.load(f'model/{target}.pkl')
# %%
for i in [1, 15, 30]:
    for key, item in regression_models.items():

        mean, std = pandas.DataFrame(item['estimators'][i].cv_results_).sort_values(
            'rank_test_r2')[['mean_test_r2', 'std_test_r2']][:1].values[0]
        print(f'------{key}-------')
        print(f'{i} -', f'{mean:.05f} +- {std:.05f}')

# %%
# Testar importância das features
ImportancesDataFrame = pandas.DataFrame()
for i in [1, 15, 30]:
    for target in targets_models:
        importances = permutation_importance(
            regression_models[target]['best_estimator'],
            regression_models[target]['windowed_data'],
            regression_models[target]['windowed_data'][target.split('-')[0]],
            n_repeats=10,
            random_state=0,
            n_jobs=-1)
        ImportancesDataFrame[f'{target}_{i}-mean'] = importances.importances_mean
        ImportancesDataFrame[f'{target}_{i}-std'] = importances.importances_std
# %%
# Printar importância das features

ImportancesDataFrame.filter(like='mean', axis='columns')

# %%
ImportancesDataFrame.set_index(
    regression_models['streamflow-s_d01']['windowed_data'].columns
).filter(like='mean', axis='columns').iloc[:9].transpose()
