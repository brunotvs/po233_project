# %%
from typing import Tuple, Union
from source.project_utils.constants import targets_models
import pandas
import matplotlib.pyplot as plt
import joblib
from matplotlib.transforms import Affine2D

pandas.set_option('display.max_columns', 51)

plt.style.use('default')

idx = pandas.IndexSlice

# %%
# Carregar o modelo
debug = False
models_path = 'model/'
if debug:
    models_path += 'debug/'

# %%
regression_models = {}
targets_models = range(1, 30, 7)
for target in targets_models:
    regression_models[target] = joblib.load(models_path + f'streamflow-s_d{target:02d}.pkl')

# %%
ImportancesDataFrame = joblib.load(models_path + 'importances.pkl')

#
# %%


def cm_to_inches(cm):
    return cm / 2.54


def column_label(column: Tuple[str, Union[str, int]]):
    label = column[0].replace('_', ' ').capitalize()
    if column[1] != "":
        label += f' River {column[1]:02d}'

    return label


def plot_save_importance(
        windowing=1,
        lim=None,
        legend_loc=0,
        save_folder: str = '.',
        x_label=None,
        y_label=None,
        width=6.4,
        height=4.8):
    if not save_folder.endswith('/'):
        save_folder += '/'
    aaaa = ImportancesDataFrame.copy().transpose()
    aaaa[('mean', windowing, 'mean')] = [row.mean()
                                         for row in ImportancesDataFrame.transpose().loc[idx[:, :, :], idx[:, windowing, 'mean']].values]
    aaaa = aaaa.sort_values(('mean', windowing, 'mean'), ascending=False)[:5]
    aaaa = aaaa.drop(columns=('mean', windowing, 'mean')).transpose()
    for column in aaaa.loc[idx[:, windowing, 'mean']].columns:

        plt.plot(
            aaaa.loc[idx[:, windowing, 'mean']][column].index,
            aaaa.loc[idx[:, windowing, 'mean']][column],
            label=column_label(column),
        )

    plt.ylim(lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(ImportancesDataFrame.loc[idx[:, windowing, 'mean']][column[0]].index)

    if legend_loc is not None:
        plt.legend(loc=legend_loc, bbox_to_anchor=(1.05, 1))
    plt.gcf().set_size_inches(cm_to_inches(width), cm_to_inches(height))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_folder + f'Importance_w{windowing:02d}.svg')


# %%
width = 15
height = 5

# %%
plot_save_importance(windowing=1,
                     legend_loc='upper left',
                     x_label='Shift (days)',
                     y_label='Importance',
                     width=width,
                     height=height,
                     save_folder='paper/graphs')

# %%
plot_save_importance(windowing=15,
                     legend_loc='upper left',
                     x_label='Shift (days)',
                     y_label='Importance',
                     width=width,
                     height=height,
                     save_folder='paper/graphs')

# %%
plot_save_importance(windowing=30,
                     legend_loc='upper left',
                     x_label='Shift (days)',
                     y_label='Importance',
                     width=width,
                     height=height,
                     save_folder='paper/graphs')

# %%
indexes = []
for windowing in [1, 15, 30]:
    for lagging in range(1, 30, 7):
        indexes.append((windowing, lagging))

multiIndex = pandas.MultiIndex.from_tuples(indexes, names=['windowing', 'lagging'])

# %%
columns = []
scores = {
    'MAE': {
        'mean': 'mean_test_neg_mean_absolute_error',
        'std': 'std_test_neg_mean_absolute_error',
        'rank': 'rank_test_neg_mean_absolute_error'
    },
    'R2': {
        'mean': 'mean_test_r2',
        'std': 'std_test_r2',
        'rank': 'rank_test_r2'
    },
}

set_estimators = {
    reg for reg in regression_models[target]['estimators'][1].cv_results_['param_reg']}
estimators = {}
for x in set_estimators:
    try:
        name = x.regressor.__class__.__name__
    except AttributeError:
        name = x.__class__.__name__

    estimators[name] = x
for regressor_name, regressor in estimators.items():
    for score_name, score in scores.items():
        for name, column_name in score.items():
            columns.append((regressor_name, score_name, name))

multiColumns = pandas.MultiIndex.from_tuples(columns, names=['regressor', 'score', 'value'])

# %%
ScoresDataFrame = pandas.DataFrame(index=multiIndex, columns=multiColumns)
for windowing, lagging in multiIndex:
    for regressor, score, statistic in multiColumns:

        UniqueRegressorsDataFrame = pandas.DataFrame(
            regression_models[lagging]['estimators'][windowing].cv_results_
        ).\
            sort_values(scores[score]['rank']).\
            drop_duplicates('param_reg')

        names = []
        for estimator in UniqueRegressorsDataFrame['param_reg']:
            try:
                names.append(estimator.regressor.__class__.__name__)
            except AttributeError:
                names.append(estimator.__class__.__name__)

        UniqueRegressorsDataFrame['name'] = names

        ScoresDataFrame.loc[(windowing, lagging), (regressor, score, statistic)] = \
            UniqueRegressorsDataFrame[UniqueRegressorsDataFrame['name'] == regressor][scores[score][statistic]].values[0]


# %%
# função para plotar


def plot_save_score(
        windowing=1,
        scorer='R2',
        lim=None,
        legend_loc=0,
        save_folder: str = '.',
        x_label=None,
        y_label=None,
        width=6.4,
        height=4.8):
    if not save_folder.endswith('/'):
        save_folder += '/'

    _, ax = plt.subplots()

    regressors = ScoresDataFrame.loc[idx[windowing], idx[:, scorer]].columns.unique('regressor')
    n_plots = len(regressors)
    k = -0.2 * n_plots / 2
    for reg in regressors:
        trans1 = Affine2D().translate(k, 0.0) + ax.transData
        plt.errorbar(
            ScoresDataFrame.loc[windowing][(reg, scorer, 'mean')].index,
            ScoresDataFrame.loc[windowing][(reg, scorer, 'mean')],
            ScoresDataFrame.loc[windowing][(reg, scorer, 'std')],
            label=reg,
            transform=trans1
        )
        k += 0.2
    plt.ylim(lim)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(ScoresDataFrame.loc[windowing][(reg, scorer, 'mean')].index)
    if legend_loc is not None:
        plt.legend(loc=legend_loc, bbox_to_anchor=(1.05, 1))
    plt.gcf().set_size_inches(cm_to_inches(width), cm_to_inches(height))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_folder + f'{scorer}_w{windowing:02d}.svg')


# %%
width = 7.3
# %%
plot_save_score(windowing=1,
                scorer='R2',
                lim=(0.5, 1),
                legend_loc=None,
                x_label='Shift (days)',
                y_label='$R^2$ score',
                width=width,
                height=height,
                save_folder='paper/graphs')

# %%
plot_save_score(windowing=15,
                scorer='R2',
                lim=(0.5, 1),
                legend_loc=None,
                x_label='Shift (days)',
                y_label='$R^2$ score',
                width=width,
                height=height,
                save_folder='paper/graphs')

# %%
plot_save_score(windowing=30,
                scorer='R2',
                lim=(0.5, 1),
                legend_loc='upper left',
                x_label='Shift (days)',
                y_label='$R^2$ score',
                width=width + 6.5,
                height=height,
                save_folder='paper/graphs')

# %%
plot_save_score(windowing=1,
                scorer='MAE',
                lim=(-300, -80),
                legend_loc=None,
                x_label='Shift (days)',
                y_label='MAE score',
                width=width,
                height=height,
                save_folder='paper/graphs')

# %%
plot_save_score(windowing=15,
                scorer='MAE',
                lim=(-300, -80),
                legend_loc=None,
                x_label='Shift (days)',
                y_label='MAE score',
                width=width,
                height=height,
                save_folder='paper/graphs')

# %%
plot_save_score(windowing=30,
                scorer='MAE',
                lim=(-300, -80),
                legend_loc='upper left',
                x_label='Shift (days)',
                y_label='MAE score',
                width=width + 6.5,
                height=height,
                save_folder='paper/graphs')

# %%
