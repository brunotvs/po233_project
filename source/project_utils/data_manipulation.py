from typing import List

import pandas
from pandas import DataFrame
from pandas.core.indexes.multi import MultiIndex


def generate_aggregation(fun, *args) -> dict:

    dataframe_column_indexes = _generate_combinations(*args)

    aggregation = {}
    for column_index in dataframe_column_indexes:
        aggregation[column_index] = fun

    return aggregation


def _generate_combinations(*args) -> List[tuple]:

    iterator = _make_iterator(args[0])

    if len(args) == 1:
        return [(name,) for name in iterator]

    combination = []
    for column_name in iterator:
        children_columns = _generate_combinations(*args[1:])
        for child_column_name in children_columns:
            combination.append((column_name,) + child_column_name)

    return combination


def _make_iterator(arg):
    if not isinstance(arg, str):
        try:
            iterator = iter(arg)
        except BaseException:
            iterator = iter([arg])
    else:
        iterator = [arg]

    return iterator


if __name__ == '__main__':
    agg = generate_aggregation('sum', 'ab', range(2))

    print(agg)


class ColumnsLoc():

    def __init__(self, *args) -> None:
        self.args = args

    def get_loc(self, df: DataFrame):
        columns = df.columns

        idx = []
        for col in self.args:
            try:
                idx += list(columns.get_locs([col]))
            except AttributeError:
                idx += [columns.get_loc(col)]

        return idx


# def get_loc():
#     ColumnsLoc().get_loc()
