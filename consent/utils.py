import json
from typing import Union
import numpy as np
import pandas as pd
import tensorflow as tf


def load_data(file_path, code_name):
    try:
        data_df = pd.read_csv(file_path, index_col=0)
    except:
        data_df = pd.read_csv(file_path, index_col=0, sep='|')

    # Requires columns = username, text/message, group/dialog_id, <code>
    data_df = data_df.rename(columns={
        'message': 'text',
        'group': 'dialog_id',
        code_name: 'code'
    })
    data_df['text'] = data_df['text'].fillna(' ')\
                                     .astype(str)
    return data_df


def train_test_split(data_df: pd.DataFrame,
                     test_size: float,
                     random_state: int = 0):
    assert 0 < test_size < 1, "Please refer to a percentage."
    np.random.seed(random_state)
    test_groups = int(data_df['dialog_id'].nunique() * test_size)
    test_groups = np.random.choice(data_df['dialog_id'].unique(), test_groups)
    train = data_df.loc[~data_df['dialog_id'].isin(test_groups)]
    test = data_df.loc[data_df['dialog_id'].isin(test_groups)]
    return train, test


def train_val_sampler(texts, contexts, labels,
                      limit_training_samples: int = -1,
                      val_size: float = 0.1,
                      batch_size: int = 64,
                      random_state: int = 0):
    np.random.seed(random_state)
    rng = np.random.default_rng()
    ix = rng.permutation(labels.shape[0])
    if limit_training_samples == -1:
        n_samples = labels.shape[0]
    else:
        n_samples = limit_training_samples
    ix = rng.choice(ix, n_samples,
                    replace=False, shuffle=False)
    val_size = int(val_size * len(ix))

    def data_generator(texts_inputs, contexts_inputs, labels_outputs):
        for t, c, l in zip(texts_inputs, contexts_inputs, labels_outputs):
            yield {"text_input": t, "con_input": c}, \
                  {"sent_output": l, "consent_output": l}
    output_types = ({"text_input": tf.string, "con_input": np.float32},
                    {"sent_output": np.float32, "consent_output": tf.float32})

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=output_types,
        args=[texts[ix], contexts[ix], labels[ix]]
    )
    val_dataset = dataset.take(val_size).batch(batch_size)
    train_dataset = dataset.skip(val_size).batch(batch_size)
    return train_dataset, val_dataset


def convert(d):
    """
    Converts dict `d` that has numpy types to Python native types.
    """
    if isinstance(d, dict):
        for k in d.keys():
            if type(d[k])==np.array:
                if type(d[k][0])==np.int32 or type(d[k][0])==np.int64:
                    d[k]=np.array(d[k],dtype=int)
                elif type(d[k][0])==np.float32 or type(d[k][0])==np.float64:
                    d[k]=np.array(d[k],dtype=float)
