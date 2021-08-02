import numpy as np

from sklearn.model_selection import KFold, train_test_split
from dataclasses import dataclass
from functools import partial
from itertools import groupby

@dataclass(eq=False)
class RunBalancedKFold:
    n_splits: int
    random_state: int
    val_size: float
        
    def split(self, sxxs, ship_runs):
        sample_indices = np.arange(0, sxxs.shape[0])
        separate_indices = list(self.groupby(sample_indices, ship_runs))
        run_cvos = list(map(KFold(n_splits=self.n_splits).split, separate_indices))
        while cvo_stack := list(map(next, run_cvos)):
            train_stack, test_stack = zip(*cvo_stack)
            X_train_stack, X_test_stack = zip(*map(self._apply_cv_mask, train_stack, test_stack, separate_indices))
            # y_train_stack, y_test_stack = zip(*map(self._apply_cv_mask, train_stack, test_stack, ship_trgts))

            # X_train_stack, X_val_stack, y_train_stack, y_val_stack = zip(*map(self._train_val_split, X_train_stack, y_train_stack))
            X_train_stack, X_val_stack = zip(*map(self._train_val_split, X_train_stack))

            # X_train, y_train = self._unstack(X_train_stack, y_train_stack)
            # X_val, y_val = self._unstack(X_val_stack, y_val_stack)
            # X_test, y_test = self._unstack(X_test_stack, y_test_stack)
            X_train = np.concatenate(X_train_stack, axis=0)
            X_val   = np.concatenate(X_val_stack, axis=0)
            X_test  = np.concatenate(X_test_stack, axis=0)

            # yield (X_train, y_train,), (X_val, y_val,), (X_test, y_test,)
            yield X_train, X_val, X_test

    @staticmethod
    def groupby(data, groups):
        for key in np.unique(groups):
            yield data[groups == key]

    @staticmethod
    def _apply_cv_mask(train, test, data):
        return data[train], data[test]

    @staticmethod
    def _unstack(x, y):
        return np.concatenate(x, axis=0), np.concatenate(y, axis=0)
    
    def _train_val_split(self, *args):
        train_val_split = partial(train_test_split, test_size=self.val_size, random_state=self.random_state)
        return train_val_split(*args)
      