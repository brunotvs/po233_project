# %%
import eli5
import joblib
import numpy
import pandas
from eli5.sklearn import PermutationImportance
from IPython import get_ipython
from mlxtend.feature_selection import ColumnSelector
from scipy.stats.stats import DescribeResult
from sklearn.compose import (ColumnTransformer, TransformedTargetRegressor,
                             make_column_selector)
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
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sqlalchemy import select
from sqlalchemy.sql.expression import true

from source.custom_transfomers.dataframe_transformers import (
    AggregateTransformer,
    GroupByTransformer,
    PivotTransformer,
    ResetIndexTransformer,
    TestTargetColumnSelector,
    TestTransformer)
from source.custom_transfomers.debug_transformer import Debug, DebugY
from source.custom_transfomers.time_window_transformer import \
    TimeWindowTransformer
from source.data_base.connection import session
from source.data_base.models import models
from source.project_utils.constants import targets
from source.project_utils.data_manipulation import (ColumnsLoc,
                                                    generate_aggregation)

pandas.set_option('display.max_columns', 51)

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# # %%
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# %%
# Construção do dataframe utilizando buscas no banco de dados sql
print('SQL query...')
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
    join(models.Reservoir, models.Variables.date == models.Reservoir.date).\
    order_by(models.Variables.date)

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

months = [date.month for date in ConsolidatedDataFrame.index.get_level_values('date')]
ConsolidatedDataFrame.insert(0, 'month', months)

ConsolidatedDataFrame.insert(
    loc=ConsolidatedDataFrame.columns.get_loc(('level', '')) + 1,
    column='yesterday_level',
    value=pandas.DataFrame(ConsolidatedDataFrame['level']).shift(1).values
)

ConsolidatedDataFrame.insert(
    loc=ConsolidatedDataFrame.columns.get_loc(('streamflow', '')) + 1,
    column='yesterday_streamflow',
    value=pandas.DataFrame(ConsolidatedDataFrame['streamflow']).shift(1).values
)

ConsolidatedDataFrame = ConsolidatedDataFrame.dropna()

ConsolidatedDataFrame.insert(
    loc=ConsolidatedDataFrame.columns.get_loc(('level', '')) + 2,
    column='level_variation',
    value=ConsolidatedDataFrame.level - ConsolidatedDataFrame.yesterday_level
)

ConsolidatedDataFrame.insert(
    loc=ConsolidatedDataFrame.columns.get_loc(('streamflow', '')) + 2,
    column='streamflow_variation',
    value=ConsolidatedDataFrame.streamflow - ConsolidatedDataFrame.yesterday_streamflow
)


# %%
seed = 0
scorer = make_scorer(accuracy_score)

rivers = session.query(models.River).all()

var_cols = ['precipitation', 'evaporation', 'temperature', 'surface_runoff']

date_cols = ['month']

shifted_cols = ['shifted_level', 'shifted_streamflow']

# %%
cv = KFold(n_splits=10, random_state=seed, shuffle=True)


pipeline_steps = [
    ('columns', ColumnTransformer([
        (
            'normalize',
            MinMaxScaler(feature_range=(0, 1)),
            ColumnsLoc(var_cols).get_loc
        ),
        (
            'month',
            OrdinalEncoder(),
            ColumnsLoc(date_cols).get_loc
        ),
        (
            'historical',
            'passthrough',
            ColumnsLoc(shifted_cols).get_loc
        )
    ],
        remainder='drop',
    )),
    ('reg', DummyRegressor())
]

grid_search_params = dict(
    param_grid=[
        # Para testes rápidos
        {
            'reg': [DecisionTreeRegressor(max_depth=5)]
        },
        {
            'reg': [
                TransformedTargetRegressor(
                    transformer=MinMaxScaler(feature_range=(0, 1)),
                    regressor=SVR(cache_size=1000)
                )
            ],
            'reg__regressor__C': range(15, 31, 5),
            'reg__regressor__gamma': ['auto', 'scale'],
            'reg__regressor__kernel': ['rbf']
        },
        {
            'reg': [
                TransformedTargetRegressor(
                    regressor=RandomForestRegressor()
                )
            ],
            'reg__transformer': [
                MinMaxScaler(feature_range=(0, 1)),
                None
            ],
            'reg__regressor__random_state': [seed],
            'reg__regressor__n_estimators': [10, 100, 250],
            'columns__normalize': [
                MinMaxScaler(feature_range=(0, 1)),
                'passthrough'
            ],
        },
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
            'reg__RandomForest__n_estimators': [10, 100, 250],
            'reg__SVR__regressor__C': range(15, 31, 5),
        },
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
        #     'reg__RandomForest__n_estimators': [10, 100, 1000],
        #     'reg__SVR__regressor__C': range(15, 31, 5),
        #     'reg__final_estimator__regressor__C': range(7, 16, 5),
        # },
    ],
    scoring=[
        'explained_variance',
        'max_error',
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
        'neg_root_mean_squared_error',
        # 'neg_mean_squared_log_error',
        'neg_median_absolute_error',
        'r2',
        # 'neg_mean_poisson_deviance',
        # 'neg_mean_gamma_deviance',
        'neg_mean_absolute_percentage_error'
    ],
    cv=cv,
    n_jobs=-1,
    verbose=2,
    error_score="raise",
    refit='r2'
)

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
print('GridSearch...')
shift_regressors = {}
for shift in [1, 8, 15, 22, 29, 30]:
    ShiftedDataFrame = ConsolidatedDataFrame.copy()

    ShiftedDataFrame.insert(
        loc=ShiftedDataFrame.columns.get_loc(('level', '')) + 1,
        column='shifted_level',
        value=pandas.DataFrame(ShiftedDataFrame['level']).shift(shift).values
    )

    ShiftedDataFrame.insert(
        loc=ShiftedDataFrame.columns.get_loc(('streamflow', '')) + 1,
        column='shifted_streamflow',
        value=pandas.DataFrame(ShiftedDataFrame['streamflow']).shift(shift).values
    )

    ShiftedDataFrame = ShiftedDataFrame.dropna()

    ShiftedDataFrame['streamflow'] = ShiftedDataFrame['streamflow'].clip(0)

    shift_regressors[shift] = {}
    target_regressor = shift_regressors[shift]
    for target in ['streamflow']:  # targets:
        target_regressor[target] = dict(
            best_score=-9999,
            estimators={},
        )
        for roll in [1, 15, 30]:
            search = GridSearchCV(Pipeline(steps=pipeline_steps), **grid_search_params)
            WindowedDataFrame = TimeWindowTransformer(var_cols, roll, agg, True).fit_transform(ShiftedDataFrame)
            target_regressor[target]['estimators'][roll] = search.fit(WindowedDataFrame, WindowedDataFrame[target])
            if search.best_score_ > target_regressor[target]['best_score']:
                target_regressor[target]['best_score'] = search.best_score_
                target_regressor[target]['best_params'] = search.best_params_
                target_regressor[target]['best_estimator'] = search.best_estimator_
                target_regressor[target]['best_windowing'] = roll
                target_regressor[target]['windowed_data'] = WindowedDataFrame

        joblib.dump(target_regressor[target], f'model/{target}-s_d{shift:02d}.pkl')

# %%
