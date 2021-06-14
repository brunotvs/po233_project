# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor
from custom_transfomers.date_window import TimeWindowTransformer
from sklearn.dummy import DummyRegressor
from project_utils.data_manipulation import generate_aggregation
from sklearn.metrics import make_scorer
import pandas
from data_base.connection import session
from data_base.models import models
from sqlalchemy import select
from IPython import get_ipython

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


# %%
RawDataFrame = pandas.read_sql(query, session.bind)
RawDataFrame


# %%
# DataFrame consolidado porém com os atributos para cada rio posicionados em uma diferente coluna
ConsolidatedDataFrame = (
    RawDataFrame.
    groupby(['date', 'river', 'level', 'streamflow']).
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
ConsolidatedDataFrame


# %%
(ConsolidatedDataFrame - ConsolidatedDataFrame.min()) / (ConsolidatedDataFrame.max() - ConsolidatedDataFrame.min())


# %%

seed = 0
scorer = make_scorer(accuracy_score)


# %%
rivers = session.query(models.River).all()

precipitation_agg = generate_aggregation('sum', 'precipitation', [river.id for river in rivers])
evaporation_agg = generate_aggregation('sum', 'evaporation', [river.id for river in rivers])
temperature_agg = generate_aggregation('mean', 'temperature', [river.id for river in rivers])
runoff_agg = generate_aggregation('mean', 'surface_runoff', [river.id for river in rivers])

cols = ['precipitation', 'evaporation', 'temperature', 'surface_runoff']


# %%
agg = precipitation_agg
agg.update(evaporation_agg)
agg.update(temperature_agg)
agg.update(runoff_agg)


# %%


# %%
clf_search = GridSearchCV(
    Pipeline([
        ('windowing', TimeWindowTransformer(columns=cols)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('clf', DummyRegressor())
    ]),
    param_grid=[
        {
            'windowing__aggregate': [agg],
            'windowing__rolling': range(1, 30, 5),
            'windowing__dropna': [False],
            'clf': (
                TransformedTargetRegressor(
                    transformer=MinMaxScaler(feature_range=(0, 1)),
                    regressor=SVR(cache_size=1000)
                ),),
            'clf__regressor__C': range(1, 15, 3),
            'clf__regressor__gamma': ['auto', 'scale'],
            'clf__regressor__kernel': ['rbf']
        },
        {
            'windowing__aggregate': [agg],
            'windowing__rolling': range(1, 30, 5),
            'windowing__dropna': [False],
            'clf': (RandomForestRegressor(), ),
            'clf__random_state': [seed],
            'clf__n_estimators': [200]
        },
        {
            'windowing__aggregate': [agg],
            'windowing__rolling': range(1, 30, 5),
            'windowing__dropna': [False],
            'clf': (DecisionTreeRegressor(), ),
            'clf__random_state': [seed]
        }
    ],
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=10,
    error_score='raise'
)


# %%
classifier = clf_search

df = pandas.DataFrame()
df['level'] = ConsolidatedDataFrame.index.get_level_values('level')
classifier.fit(ConsolidatedDataFrame, ConsolidatedDataFrame.index.get_level_values('level'))
df['p_level'] = classifier.predict(ConsolidatedDataFrame)
classifier.fit(ConsolidatedDataFrame, ConsolidatedDataFrame.index.get_level_values('streamflow'))
df['streamflow'] = ConsolidatedDataFrame.index.get_level_values('streamflow')
df['p_streamflow'] = classifier.predict(ConsolidatedDataFrame)
df
