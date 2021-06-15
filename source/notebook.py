# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# %%
import pandas
from IPython import get_ipython
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

from custom_transfomers.date_window import TimeWindowTransformer
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
grid_search_params = dict(
    estimator=Pipeline([
        ('windowing', TimeWindowTransformer(columns=cols)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('clf', DummyRegressor())
    ]),
    param_grid=[
        {
            'windowing__aggregate': [agg],
            'windowing__rolling': range(1, 32, 10),
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
            'windowing__rolling': range(1, 32, 10),
            'windowing__dropna': [False],
            'clf': (RandomForestRegressor(), ),
            'clf__random_state': [seed],
            'clf__n_estimators': [200]
        },
        {
            'windowing__aggregate': [agg],
            'windowing__rolling': range(1, 32, 10),
            'windowing__dropna': [False],
            'clf': (DecisionTreeRegressor(), ),
            'clf__random_state': [seed]
        },
        {
            'windowing__aggregate': [agg],
            'windowing__rolling': range(1, 32, 10),
            'windowing__dropna': [False],
            'clf': (StackingRegressor(
                estimators=[('RandomForest', RandomForestRegressor()), ('SVR', SVR())],
                final_estimator=Ridge()
            ),),
            'clf__RandomForest__random_state': [seed],
            'clf__RandomForest__n_estimators': [200],
        }
    ],
    scoring='neg_root_mean_squared_error',
    cv=10,
    n_jobs=-1,
    verbose=10,
    error_score='raise'
)


# %%
targets = ['level', 'streamflow']

clf_search = {target: GridSearchCV(**grid_search_params) for target in targets}


# %%
streamflow_X_train, streamflow_X_test, streamflow_y_train, streamflow_y_test = train_test_split(
    ConsolidatedDataFrame,
    ConsolidatedDataFrame.index.get_level_values('streamflow'), random_state=seed
)

level_X_train, level_X_test, level_y_train, level_y_test = train_test_split(
    ConsolidatedDataFrame,
    ConsolidatedDataFrame.index.get_level_values('level'), random_state=seed
)


# %%
level_classifier = clf_search['level']
streamflow_classifier = clf_search['streamflow']

streamflow_classifier.fit(streamflow_X_train, streamflow_y_train)
level_classifier.fit(level_X_train, level_y_train)


# %%
df = pandas.DataFrame()
df['level'] = level_y_test
df['p_level'] = level_classifier.predict(level_X_test)
df['streamflow'] = streamflow_y_test
df['p_streamflow'] = streamflow_classifier.predict(streamflow_X_test)


# %%
level_results = pandas.DataFrame(level_classifier.cv_results_).sort_values('rank_test_score', ascending=False)
{result['param_clf']: (result['mean_test_score'], result['std_test_score']) for _, result in level_results.iterrows()}


# %%
streamflow_results = pandas.DataFrame(streamflow_classifier.cv_results_).sort_values('rank_test_score', ascending=False)
{result['param_clf']: (result['mean_test_score'], result['std_test_score'])
 for _, result in streamflow_results.iterrows()}


# %%
df
