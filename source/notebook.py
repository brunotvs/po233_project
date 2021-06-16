# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas
from IPython import get_ipython
from sklearn import tree
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sqlalchemy import select

from custom_transfomers.date_window import TimeWindowTransformer, Debug
from data_base.connection import session
from data_base.models import models
from project_utils.data_manipulation import generate_aggregation


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
    pivot(index=["date", 'level', 'streamflow'], columns="river")
)

ConsolidatedDataFrame.insert(0, 'previous_streamflow', pandas.DataFrame(
    ConsolidatedDataFrame.index.get_level_values('streamflow')).shift(1).values)
ConsolidatedDataFrame.insert(0, 'previous_level', pandas.DataFrame(
    ConsolidatedDataFrame.index.get_level_values('level')).shift(1).values)

ConsolidatedDataFrame = ConsolidatedDataFrame.dropna()


# %%
seed = 0
scorer = make_scorer(accuracy_score)

rivers = session.query(models.River).all()

precipitation_agg = generate_aggregation('sum', 'precipitation', [river.id for river in rivers])
evaporation_agg = generate_aggregation('sum', 'evaporation', [river.id for river in rivers])
temperature_agg = generate_aggregation('mean', 'temperature', [river.id for river in rivers])
runoff_agg = generate_aggregation('mean', 'surface_runoff', [river.id for river in rivers])

cols = ['precipitation', 'evaporation', 'temperature', 'surface_runoff']

agg = precipitation_agg
agg.update(evaporation_agg)
agg.update(temperature_agg)
agg.update(runoff_agg)

# %%
cv = KFold(n_splits=10, random_state=seed, shuffle=True)

windowing_params = {
    'windowing__aggregate': [agg],
    'windowing__rolling': range(1, 32, 10),
    'windowing__dropna': [False],
}

grid_search_params = dict(
    estimator=Pipeline(
        [
            ('windowing', TimeWindowTransformer(columns=cols)),
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler(feature_range=(0, 1))),
            ('clf', DummyRegressor())
        ]
    ),
    param_grid=[
        {
            **windowing_params,
            'clf': [TransformedTargetRegressor(
                regressor=SVR(cache_size=1000)
            )],
            'clf__transformer': [MinMaxScaler(feature_range=(0, 1))],
            'clf__regressor__C': range(1, 15, 3),
            'clf__regressor__gamma': ['auto', 'scale'],
            'clf__regressor__kernel': ['rbf']
        },
        {
            **windowing_params,
            'scaler': [MinMaxScaler(), None],
            'clf': [TransformedTargetRegressor(
                regressor=RandomForestRegressor()
            )],
            'clf__transformer': [MinMaxScaler(feature_range=(0, 1)), None],
            'clf__regressor__random_state': [seed],
            'clf__regressor__n_estimators': [200],
        },
        {
            **windowing_params,
            'clf': (DecisionTreeRegressor(), ),
            'clf__random_state': [seed]
        },
        {
            **windowing_params,
            'clf': (StackingRegressor(
                estimators=[
                    ('RandomForest', RandomForestRegressor()),
                    ('SVR', TransformedTargetRegressor(
                        transformer=MinMaxScaler(feature_range=(0, 1)),
                        regressor=SVR()
                    ))
                ],
                final_estimator=TransformedTargetRegressor(
                    transformer=MinMaxScaler(feature_range=(0, 1)),
                    regressor=SVR()
                )
            ),),
            'clf__RandomForest__random_state': [seed],
            'clf__RandomForest__n_estimators': [200],
            'clf__SVR__regressor__C': range(1, 16, 5),
            'clf__final_estimator__regressor__C': range(1, 16, 5)
        }
    ],
    scoring='r2',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    error_score='raise'
)


# %%
targets = ['level', 'streamflow']

clf_search = {target: GridSearchCV(**grid_search_params) for target in targets}

clf_search['level'].fit(ConsolidatedDataFrame, ConsolidatedDataFrame.index.get_level_values('level'))
clf_search['streamflow'].fit(ConsolidatedDataFrame, ConsolidatedDataFrame.index.get_level_values('streamflow'))


# %%
f"{clf_search['level'].best_score_} ± {clf_search['level'].cv_results_['std_test_score'][clf_search['level'].best_index_]}"

# %%
f"{clf_search['streamflow'].best_score_} ± {clf_search['streamflow'].cv_results_['std_test_score'][clf_search['streamflow'].best_index_]}"


# %%
level_estimator = clf_search['level'].best_estimator_

# %%
streamflow_estimator = clf_search['streamflow'].best_estimator_

# %%
full_df = pandas.DataFrame()
full_df['level'] = ConsolidatedDataFrame.index.get_level_values('level')
full_df['p_level'] = level_estimator.predict(ConsolidatedDataFrame)
full_df['streamflow'] = ConsolidatedDataFrame.index.get_level_values('streamflow')
full_df['p_streamflow'] = streamflow_estimator.predict(ConsolidatedDataFrame)
full_df


# %%
