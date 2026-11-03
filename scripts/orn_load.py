"""Load Yang's ORN trial-mean response pickles with an unpickler that only allows pandas/numpy reconstruction."""
import pickle, warnings
import pandas as pd

OK_MOD = ('numpy', 'pandas')
OK_BUILTIN = {'slice'}


class _Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] in OK_MOD or (module == 'builtins' and name in OK_BUILTIN):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')


def load(path):
    with open(path, 'rb') as f, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _Safe(f).load()


FILES = {'megamat': '/mnt/project-files/uploads/hearth/66845d40-f275-4eae-bc30-f430169da336',
         'validation2': '/mnt/project-files/uploads/hearth/e58c5dba-af61-47a5-8689-8b38d23c1b71'}
