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
query = select(
    models.Variables.date,
    models.Variables.precipitation.label('precipitation'),
    models.Variables.temperature.label('temperature'),
    models.Variables.evaporation.label('evaporation'),
    models.Variables.surface_runoff.label('surface_runoff'),
    models.Coordinate.river_id.label('river'),
    models.Reservoir.level,
    models.Reservoir.streamflow
).\
    join(models.Variables.coordinate).\
    join(models.Reservoir, models.Variables.date == models.Reservoir.date)

RawDataFrame = pandas.read_sql(query, session.bind)


# %%
# DataFrame consolidado porém com os atributos para cada rio posicionados em uma diferente coluna
ConsolidatedDataFrame = (
    RawDataFrame.
    groupby(['date', 'level', 'river', 'streamflow']).
    agg({
        'precipitation': 'sum',
        'evaporation': 'sum',
        'temperature': 'mean',
        'surface_runoff': 'mean',
    }).
    reset_index().
    pivot(index=["date", 'level', 'streamflow'], columns="river").
    reset_index(['level', 'streamflow'])
)


ConsolidatedDataFrame.insert(
    loc=ConsolidatedDataFrame.columns.get_loc(('level', '')) + 1,
    column='previous_level',
    value=pandas.DataFrame(ConsolidatedDataFrame['level']).shift(1).values
)

ConsolidatedDataFrame.insert(
    loc=ConsolidatedDataFrame.columns.get_loc(('streamflow', '')) + 1,
    column='previous_streamflow',
    value=pandas.DataFrame(ConsolidatedDataFrame['streamflow']).shift(1).values
)

ConsolidatedDataFrame = ConsolidatedDataFrame.dropna()

ConsolidatedDataFrame.insert(
    loc=ConsolidatedDataFrame.columns.get_loc(('level', '')) + 2,
    column='level_variation',
    value=ConsolidatedDataFrame.level - ConsolidatedDataFrame.previous_level
)

ConsolidatedDataFrame.insert(
    loc=ConsolidatedDataFrame.columns.get_loc(('streamflow', '')) + 2,
    column='streamflow_variation',
    value=ConsolidatedDataFrame.streamflow - ConsolidatedDataFrame.previous_streamflow
)

months = [date.month for date in ConsolidatedDataFrame.index.get_level_values('date')]
ConsolidatedDataFrame.insert(0, 'month', months)


# %%
# Carregar o modelo
regression_models = {}
for target, _ in targets_models.items():
    regression_models[target] = joblib.load(f'model/{target}.pkl')

# # %%
# # Testar importância das features
# for target in targets:
#     regression_models[target]['permutation_importance'] = permutation_importance(
#         target_regressor[target]['best_estimator'],
#         target_regressor[target]['windowed_data'],
#         target_regressor[target]['windowed_data'][target],
#         n_repeats=10,
#         random_state=0,
#         n_jobs=-1)

# # %%
# # Printar importância das features
# for target in targets:
#     print(pandas.DataFrame(target_regressor[target]['permutation_importance'].importances_mean))

# %%
for key, item in regression_models.items():
    for i in [1, 15, 30]:
        mean, std = pandas.DataFrame(item['estimators'][i].cv_results_).sort_values(
            'rank_test_r2')[['mean_test_r2', 'std_test_r2']][:1].values[0]
        print(f'------{key}-------')
        print(f'{i} -', f'{mean:.05f} +- {std:.05f}')

# %%
