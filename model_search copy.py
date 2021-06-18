# %%
import eli5
import joblib
import numpy
import pandas
from eli5.sklearn import PermutationImportance
from IPython import get_ipython
from mlxtend.feature_selection import ColumnSelector
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sqlalchemy import select
from sqlalchemy.sql.expression import true

from source.custom_transfomers.debug_transformer import Debug
from source.custom_transfomers.time_window_transformer import TimeWindowTransformer
from source.data_base.connection import session
from source.data_base.models import models
from source.project_utils.data_manipulation import generate_aggregation, get_columns_loc

pandas.set_option('display.max_columns', 51)

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# Construção do dataframe utilizando buscas no banco de dados sql
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


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
    reset_index(['streamflow'])
)


# ConsolidatedDataFrame.insert(
#     loc=ConsolidatedDataFrame.columns.get_loc(('level', '')) + 1,
#     column='previous_level',
#     value=pandas.DataFrame(ConsolidatedDataFrame['level']).shift(1).values
# )

# ConsolidatedDataFrame.insert(
#     loc=ConsolidatedDataFrame.columns.get_loc(('streamflow', '')) + 1,
#     column='previous_streamflow',
#     value=pandas.DataFrame(ConsolidatedDataFrame['streamflow']).shift(1).values
# )

# ConsolidatedDataFrame = ConsolidatedDataFrame.dropna()

# ConsolidatedDataFrame.insert(
#     loc=ConsolidatedDataFrame.columns.get_loc(('level', '')) + 2,
#     column='level_variation',
#     value=ConsolidatedDataFrame.level - ConsolidatedDataFrame.previous_level
# )

# ConsolidatedDataFrame.insert(
#     loc=ConsolidatedDataFrame.columns.get_loc(('streamflow', '')) + 2,
#     column='streamflow_variation',
#     value=ConsolidatedDataFrame.streamflow - ConsolidatedDataFrame.previous_streamflow
# )

months = [date.month for date in ConsolidatedDataFrame.index.get_level_values('date')]
ConsolidatedDataFrame.insert(0, 'month', months)


# %%
seed = 0
scorer = make_scorer(accuracy_score)

rivers = session.query(models.River).all()

var_cols = ['precipitation', 'evaporation', 'temperature', 'surface_runoff']
var_cols_river = [(var, river.id) for river in rivers for var in var_cols]
var_cols_loc = [ConsolidatedDataFrame.columns.get_loc(col) for col in var_cols_river]

date_cols = ['month']
date_cols_loc = [ConsolidatedDataFrame.columns.get_loc((col, '')) for col in date_cols]

# previous_cols = ['previous_level', 'previous_streamflow']
# previous_cols_loc = [ConsolidatedDataFrame.columns.get_loc((col, '')) for col in previous_cols]

# %%
cv = KFold(n_splits=3, random_state=seed, shuffle=True)

pipeline_steps = [
    ('columns', ColumnTransformer([
        (
            'normalize',
            MinMaxScaler(feature_range=(0, 1)),
            var_cols_loc
        ),
    ],
        remainder='drop',
    )),
    ('debug', 'passthrough'),
    ('reg', DummyRegressor())
]

grid_search_params = dict(
    param_grid=[
        # {
        #     'reg': [
        #         TransformedTargetRegressor(
        #             transformer=MinMaxScaler(feature_range=(0, 1)),
        #             regressor=SVR(cache_size=1000)
        #         )
        #     ],
        #     'reg__regressor__C': range(1, 15, 3),
        #     'reg__regressor__gamma': ['auto', 'scale'],
        #     'reg__regressor__kernel': ['rbf']
        # },
        # {
        #     'reg': [
        #         TransformedTargetRegressor(
        #             regressor=RandomForestRegressor()
        #         )
        #     ],
        #     'reg__transformer': [
        #         MinMaxScaler(feature_range=(0, 1)),
        #         None
        #     ],
        #     'reg__regressor__random_state': [seed],
        #     'reg__regressor__n_estimators': [1000],
        #     'columns__normalize': ['passthrough'],
        # },
        # {
        #     'reg': [
        #         StackingRegressor(
        #             estimators=[
        #                 ('RandomForest', RandomForestRegressor()),
        #                 ('SVR', TransformedTargetRegressor(
        #                     transformer=MinMaxScaler(feature_range=(0, 1)),
        #                     regressor=SVR()
        #                 ))
        #             ],
        #             final_estimator=TransformedTargetRegressor(
        #                 transformer=MinMaxScaler(feature_range=(0, 1)),
        #                 regressor=SVR()
        #             )
        #         )
        #     ],
        #     'reg__RandomForest__random_state': [seed],
        #     'reg__RandomForest__n_estimators': [1000],
        #     'reg__SVR__regressor__C': range(1, 16, 5),
        #     'reg__final_estimator__regressor__C': range(1, 16, 5),
        # },
        {
            'reg': [
                StackingRegressor(
                    estimators=[
                        ('RandomForest', RandomForestRegressor()),
                        ('SVR', TransformedTargetRegressor(
                            transformer=MinMaxScaler(feature_range=(0, 1)),
                            regressor=SVR()
                        ))
                    ],
                    final_estimator=Ridge()
                )
            ],
            'reg__RandomForest__random_state': [seed],
            'reg__RandomForest__n_estimators': [1000],
            'reg__SVR__regressor__C': range(1, 16, 5),
        },

    ],
    scoring='neg_root_mean_squared_error',
    cv=cv,
    n_jobs=-1,
    error_score='raise',
    refit=False
)


# %%
targets = [
    'level',
    # 'streamflow',
    # 'level_variation',
    # 'streamflow_variation'
]

reg_search = {
    target: GridSearchCV(
        Pipeline(
            steps=pipeline_steps,
            # memory=f'__pipeline_cache_{target}__'
        ),
        **grid_search_params) for target in targets}

# %%
precipitation_agg = generate_aggregation('sum', 'precipitation', [river.id for river in rivers])
evaporation_agg = generate_aggregation('sum', 'evaporation', [river.id for river in rivers])
temperature_agg = generate_aggregation('sum', 'temperature', [river.id for river in rivers])
runoff_agg = generate_aggregation('sum', 'surface_runoff', [river.id for river in rivers])

agg = precipitation_agg
agg.update(evaporation_agg)
agg.update(temperature_agg)
agg.update(runoff_agg)

# %%
regressor = {}
for target in targets:
    regressor[target] = dict(
        best_windowing=1,
        best_params=None,
        best_score=-9999,
        best_estimator=reg_search[target].estimator,
        windowing_score={},
    )
    for roll in [20]:  # range(1, 12, 5):
        data = TimeWindowTransformer(var_cols, roll, agg, True).fit_transform(ConsolidatedDataFrame)
        regressor[target]['windowing_score'][roll] = reg_search[target].fit(data, data.index.get_level_values(target))

        if reg_search[target].best_score_ > regressor[target]['best_score']:
            regressor[target]['best_score'] = reg_search[target].best_score_
            regressor[target]['best_params'] = reg_search[target].best_params_
            regressor[target]['best_estimator'].set_params(**regressor[target]['best_params'])
            regressor[target]['best_windowing'] = roll

# %%
# Calcular scores do melhor estimador
for target in targets:
    WindowedDataFrame = TimeWindowTransformer(
        columns=var_cols,
        rolling=regressor[target]['best_windowing'],
        aggregate=agg,
        dropna=True).fit_transform(ConsolidatedDataFrame)

    regressor[target]['cv_score'] = cross_validate(
        regressor[target]['best_estimator'],
        WindowedDataFrame,
        WindowedDataFrame.index.get_level_values(target),
        cv=10,
        scoring=['neg_root_mean_squared_error', 'r2'],
        n_jobs=-1)

# %%
# Printar scores encontrados
for target in targets:
    for key, val in regressor[target]['cv_score'].items():
        print(f'{target} -> {key}: {val.mean()} ± {val.std()}')

    print('\n')


# %%
# Testar importância das features
for target in targets:
    WindowedDataFrame = TimeWindowTransformer(
        columns=var_cols,
        rolling=regressor[target]['best_windowing'],
        aggregate=agg,
        dropna=True).fit_transform(ConsolidatedDataFrame)

    regressor[target]['best_estimator'].fit(WindowedDataFrame, WindowedDataFrame.index.get_level_values(target))
    regressor[target]['permutation_importance'] = permutation_importance(
        regressor[target]['best_estimator'],
        WindowedDataFrame,
        WindowedDataFrame.index.get_level_values(target),
        n_repeats=10,
        random_state=0,
        n_jobs=-1)

# %%
# Printar importância das features
for target in targets:
    print(pandas.DataFrame(regressor[target]['permutation_importance'].importances_mean))

# %%
df = pandas.DataFrame()

for target in targets:
    df[target] = WindowedDataFrame.index.get_level_values(target)
    df[f'p_{target}'] = regressor[target]['best_estimator'].predict(WindowedDataFrame)
df

# %%
for target in targets:
    print(f'{target} - {regressor[target]["best_params"]} - {regressor[target]["best_windowing"]}\n\n')

# %%
# Salvar os modelos
for target in targets:
    joblib.dump(regressor[target]['best_estimator'], f'model/{target}.pkl')
