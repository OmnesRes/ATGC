import os
os.environ['OPENBLAS_NUM_THREADS'] = '10'
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import pathlib
path = pathlib.Path.cwd()

if path.stem == 'ATGC':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('ATGC')]
    import sys
    sys.path.append(str(cwd))

##load the instance and sample data
D, samples = pickle.load(open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_3' / 'sim_data.pkl', 'rb'))

context_df = pd.DataFrame(data={'contexts': D['context'], 'sample_idx': D['sample_idx']})

context_df = context_df.groupby(['sample_idx', "contexts"]).size().unstack(fill_value=0)
context_df = pd.DataFrame.from_dict({'sample_idx': context_df.index, 'context_counts': context_df.values.tolist()})
sample_df = pd.DataFrame(data={'sample_idx': np.arange(1000), 'class': samples['classes']})
sample_df = pd.merge(sample_df, context_df, on='sample_idx')
context_counts = np.stack(sample_df['context_counts'].values)

y_label = np.array(samples['classes'])

random_forest_metrics = []
logistic_metrics = []

for i in range(3):
    reg = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=10)

    idx_train, idx_test = next(StratifiedShuffleSplit(random_state=0, n_splits=1, test_size=200).split(y_label, y_label))
    y_train, y_test = y_label[idx_train], y_label[idx_test]
    context_train, context_test = context_counts[idx_train], context_counts[idx_test]
    reg.fit(context_train, y_train)
    context_test_predictions = reg.predict_proba(context_test)
    random_forest_metrics.append(sum(np.argmax(context_test_predictions, axis=-1) == y_label[idx_test]) / len(y_label[idx_test]))

    reg = LogisticRegression()
    reg.fit(context_train, y_train)
    context_test_predictions = reg.predict_proba(context_test)
    logistic_metrics.append(sum(np.argmax(context_test_predictions, axis=-1) == y_label[idx_test]) / len(y_label[idx_test]))


with open(cwd / 'figures' / 'controls' / 'samples' / 'sim_data' / 'classification' / 'experiment_3' / 'standard_metrics.pkl', 'wb') as f:
    pickle.dump([random_forest_metrics, logistic_metrics], f)