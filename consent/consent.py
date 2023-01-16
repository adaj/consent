import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

from pathlib import Path
from collections import deque
from typing import List, Union, Optional
from pydantic import BaseModel, Field

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import tensorflow_addons as tfa
import wandb
from wandb.keras import WandbCallback

import consent.utils as utils


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# Currently only from TFHub
SUPPORTED_LANGUAGE_FEATURIZERS = [
    'https://tfhub.dev/google/wiki40b-lm-nl/1',
    'https://tfhub.dev/google/wiki40b-lm-multilingual-64k/1',
    'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',
    'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/LaBSE'
]

HYPERPARAMETERS  = [
    'language_featurizer', 'sent_hl_units', 'sent_dropout',
    'consent_hl_units', 'lags', 'augmentation',
    'learning_rate', 'batch_size'
]


class Config(BaseModel):
    """
    Configuration object. All parameters have default values in case you
    do not want to use the features related to it.

    Args:
        **kwargs: Set the attributes by providing them in the args directly.

    Attributes:
        architecture (str): A label to refer to the architecture in use.
        caller (str): A label to mention the context which the model is used.
        datasets (List[str]): Labels of datasets used to train the model.
        dataset_name (str): Labels of the dataset currently in use.
        code_name (str): Code label, i.e. the column in data_df with the labels.
        codes (List[str]): Codes/labels in the data to be learned.
        default_code (str): Default code label.
        wandb_project (str): Name of the W&B project to send results.
        load_weights (str): Load weights of layers from a pretrained model.
        language_featurizer (str): URL of TF-Hub featurizer to use.
        sent_hl_units (int): Number of units in the sent hidden layer.
        sent_dropout (float): Dropout rate for the sent hidden layer.
        consent_hl_units (int): Number of units in the consent hidden layer.
        lags (int): Number of previous codes to use.
        augmentation (dict): Data augmentation configuration.
        n_samples (int): Number of samples used during training.
        max_epochs (int): Max number of epochs to train the model.
        callback_patience (int): Epochs without improvement to stop train.
        learning_rate (float): Learning rate to train the model.
        batch_size (int): Batch size to train the model.
    """

    __options__ = [
      'architecture', 'caller',
      'datasets', 'dataset_name', 'code_name', 'codes', 'default_code',
      'load_weights', 'wandb_project', 'language_featurizer',
      'sent_hl_units', 'sent_dropout', 'consent_hl_units', 'lags',
      'augmentation', 'n_samples', 'max_epochs', 'callback_patience',
      'learning_rate', 'batch_size'
    ]
    architecture: str = "consent==0.3"
    caller: str = "train"
    datasets: List[str] = []
    dataset_name: str
    code_name: str
    codes: List[str]
    default_code: str
    wandb_project: Optional[str] = None
    load_weights: Optional[str] = None
    language_featurizer: Union[str, List[str]] = \
        'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
    sent_hl_units: Union[int, List[int]] = 256
    sent_dropout: Union[float, List[float]] = 0.8,
    consent_hl_units: Union[int, List[int]] = 32
    lags: Union[int, List[int]] = 4
    augmentation: Union[dict, bool, List] = False
    n_samples: Optional[int]
    max_epochs: int = 50
    callback_patience: int = 10
    learning_rate: Union[float, List[float]] = 1e-3
    batch_size: Union[int, List[int]] = 256

    def __str__(self):
        return json.dumps({key:value \
                            for key, value in self.__dict__.items() \
                            if  not callable(key)},
                          indent=4,
                          default=utils.convert) # Cast to be JSON serializable

    def __repr__(self):
        return f"consent.Config({self.__str__()})"


class Message(BaseModel):
    dialog_id: str = Field(..., max_length=250,
                           description="Dialog id.")
    username: str = Field(..., max_length=250,
                          description="Username.")
    text: str = Field(..., description="Raw text message.")
    code: str = Optional[str]


class DialogValidator(BaseModel):
    dialog_data: List[Message]


class ContextualInformation(BaseModel):
    from_same_user: bool
    previous_codes: List[str]


class ConSent:
    """
    ConSent model interface.
    - `train` to train a model from scratch,
    - `predict_proba` to predict outputs for one single message, and
    - `predict_sequence` to predict outputs for an entire dialog.

    Args:
        config (Config): Configuration to be used.
        load (str, bool): Load a model from its folder path.
        random_state: Control the randomness of sampling and training.

    Attributes:
        config (Config): Configuration in use.
        onehot_encoder (OneHotEncoder): Sklearn's transformer in use.
        model: Keras model in use.
        cache: Stores (as dict) contextual information of many dialog_id (key).
    """

    def __init__(self,
                 config: Config = None,
                 load: str = None,
                 random_state: int = 0,
                 extra_callbacks: Union[List, None] = None):
        if config is None:
            if load is False:
                raise ValueError("It should be provided either a new `config`"+\
                                 "object or `load` a pretrained model.")
            else:
                with open(os.path.join(load, "config.json"), 'r') as f:
                    self.config = Config(**json.load(f))
        else:
            self.config = config
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        self.random_state = random_state
        if load:
            assert os.path.isdir(load), "Trained model was not found."
            self.model = tf.keras.models.load_model(load)
        else:
            self.model = None
        # Prepare inputs and labels
        self.onehot_encoder = OneHotEncoder(categories=[self.config.codes])\
                                .fit(np.array(self.config.codes).reshape(-1, 1))
        self.cache = {} # cache contextual information by dialog_id (key)
        self.wandb_run = None


    def __repr__(self):
        return f"consent.ConSent(model={self.model})"

    def make_model(self,
                   contextual_size: int,
                   output_size: int,
                   language_featurizer: str,
                   sent_hl_units: int,
                   sent_dropout: float,
                   consent_hl_units: int):
        # Ensure language_featurizer is compatible with current implementation
        assert language_featurizer in SUPPORTED_LANGUAGE_FEATURIZERS, \
                "`language_featurizer` not supported " + \
                f"(available: {SUPPORTED_LANGUAGE_FEATURIZERS})."

        initializer = tf.keras.initializers.GlorotUniform()
        # Text branch
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string,
                                           name="text_input")
        # Contextual features branch
        con_input = tf.keras.Input(shape=(contextual_size,), name="con_input")

        # SENTence encoder
        if language_featurizer == \
            'https://tfhub.dev/google/wiki40b-lm-nl/1' \
            or language_featurizer == \
            'https://tfhub.dev/google/wiki40b-lm-multilingual-64k/1':
            encoder = hub.KerasLayer(language_featurizer,
                                     name="sent_encoder",
                                     signature="word_embeddings",
                                     output_key="word_embeddings")(text_input)
            encoder = tf.keras.layers.GlobalAveragePooling1D()(encoder)
        elif language_featurizer == \
          'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3':
            encoder = hub.KerasLayer(language_featurizer,
                                     trainable=False,
                                     name="sent_encoder")(text_input)
        elif language_featurizer == \
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4':
            preprocessor = hub.KerasLayer(
                "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
            encoder_inputs = preprocessor(text_input)
            encoder = hub.KerasLayer(
              "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4",
              trainable=False,
              output_key="pooled_output")(encoder_inputs)
        elif language_featurizer in \
            ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', \
             'sentence-transformers/LaBSE']:
            from consent.experimental import SBert
            from transformers import AutoTokenizer, TFAutoModel
            tokenizer = AutoTokenizer.from_pretrained(language_featurizer)
            model = TFAutoModel.from_pretrained(language_featurizer)
            for layer in model.layers:
                layer.trainable=False
                for w in layer.weights: w._trainable=False
            encoder = SBert(tokenizer, model)(text_input)

        # sent Dense hidden layer 1
        sent_hl = tf.keras.layers.Dense(sent_hl_units,
                                        kernel_initializer=initializer,
                                        activation='relu',
                                        name='sent_hl')(encoder)
        sent_hl_dropout = tf.keras.layers.Dropout(sent_dropout)(sent_hl)

        sent_output = tf.keras.layers.Dense(output_size,
                                  kernel_initializer=initializer,
                                  activation='softmax',
                                  name="sent_output")(sent_hl_dropout)

        # Concat
        branch_concat = tf.keras.layers.concatenate([sent_output, con_input])

        # # consent Dense hidden layer
        consent_hl = tf.keras.layers.Dense(consent_hl_units,
                                           kernel_initializer=initializer,
                                           activation='relu',
                                           name="consent_hl")(branch_concat)
        consent_hl_dropout = tf.keras.layers.Dropout(.5)(consent_hl)

        # Dense final output
        consent_output = tf.keras.layers.Dense(output_size,
                                    kernel_initializer=initializer,
                                    activation='softmax',
                                    name="consent_output")(consent_hl_dropout)

        return tf.keras.Model(inputs=[text_input, con_input],
                              outputs=[sent_output, consent_output])


    def prepare_inputs(self, dialog_data: pd.DataFrame):
        """
        Extracts texts and contexts (inputs for ConSent.model) from
        dialog_data samples.
        """
        texts = dialog_data['text'].values.astype(str)
        contexts = np.concatenate([
            # 1. Contains question mark?
            dialog_data['text'].apply(lambda x: ('?' in x))\
                .astype(int).values.reshape(-1,1),
            # 2. It's from the same user?
            (dialog_data['username']==dialog_data['username'].shift())\
                .astype(int).values.reshape(-1,1),
            # 3. What were the (predicted) previous codes?
            self.extract_previous_codes_by_dialog_id(dialog_data)
        ], axis=1).astype(np.float32)
        return texts, contexts


    def extract_previous_codes_by_dialog_id(self,
                                            dialog_data: pd.DataFrame):
        """
        Extracts the previous codes of all previous codes in all dialog_ids
        available on dialog_data.
        As of current implementation, this is only useful to use within
        prepare_inputs().
        """
        def extract_lags(labels: pd.Series, default_code: str, lags: int):
            return pd.concat(
              [labels.shift(i).fillna(default_code) for i in range(1, lags+1)],
              axis=1
            )
        previous_codes = dialog_data.groupby('dialog_id')\
                                    .apply(lambda x: extract_lags(
                                        labels=x['code'],
                                        default_code=self.config.default_code,
                                        lags=self.config.lags))\
                                    .apply(self.onehot_encode, axis=1)\
                                    .apply(np.ravel)
        return np.stack(previous_codes)


    def prepare_labels(self, dialog_data: pd.DataFrame):
        """
        Transforms the labels into one hot encoded representations.
        """
        # One hot encoders to represent classes as Boolean vectors
        return self.onehot_encoder.transform(dialog_data[['code']].values)\
                   .toarray()


    def train(self,
              dialog_data: pd.DataFrame,
              limit_samples: int = -1,
              tf_verbosity: int = 2,
              save_model: str = None):
        """
        Train and save a ConSent model from scratch using `dialog_data` and
        the `config` provided on __init__.

        Args:
            dialog_data (pd.DataFrame): Data of all dialogs, with the columns
                'dialog_id', 'username', 'text', 'code'.
            limit_samples (int): Train using less samples than the total.
            tf_verbosity (int): Set the verbosity flag of tensorflow.
            save_model (str): Path to save the keras model object.
        """
        if limit_samples > 0:
            self.config.n_samples = limit_samples
            if 0 < limit_samples <= 1:
                limit_samples = int(dialog_data.shape[0] * limit_samples)
        else:
            self.config.n_samples = dialog_data.shape[0]
        # Append the dataset_name to the list of `datasets` used to train
        if self.config.dataset_name not in self.config.datasets:
            self.config.datasets.append(self.config.dataset_name)

        # Validate dialog_data
        # DialogValidator(dialog_data=dialog_data.to_dict(orient="records"))

        texts, contexts = self.prepare_inputs(dialog_data)
        labels = self.prepare_labels(dialog_data)
        # Apply data augmentation (attention: require indexed samples)
        if self.config.augmentation:
            aug_text = pd.read_csv(self.config.augmentation['data_file_path'],
                                   encoding='utf-8', sep='|', index_col=0)
            indices = dialog_data.loc[dialog_data['code']\
                                      .isin(self.config.augmentation['codes'])]\
                                 .reset_index()\
                                 .join(aug_text,
                                       on='index', lsuffix='_original')\
                                 .dropna().index
            aug_text = aug_text.loc[dialog_data.iloc[indices].index\
                                               .intersection(aug_text.index)]
            aug_text['text'] = aug_text['text'].fillna(' ')
            texts = np.concatenate([texts,
                                    aug_text['text'].values])
            contexts = np.concatenate([contexts,
                                       contexts[indices]])
            labels = np.concatenate([labels,
                                     labels[indices]])
        # Make model
        self.model = self.make_model(
            contextual_size = contexts.shape[1],
            output_size = labels.shape[1],
            language_featurizer = self.config.language_featurizer,
            sent_hl_units = self.config.sent_hl_units,
            sent_dropout = self.config.sent_dropout,
            consent_hl_units = self.config.consent_hl_units
        )
        # Apply fine tuning on a pretrained model by reusing its weights
        if self.config.load_weights:
            # Load parameters from a pretrained model
            pre_trained_model = \
                tf.keras.models.load_model(self.config.load_weights)
            sent_weights = pre_trained_model.get_layer('sent_hl')\
                                       .get_weights()
            self.model.get_layer('sent_hl')\
                        .set_weights(sent_weights)
            consent_weights = pre_trained_model.get_layer('consent_hl')\
                                       .get_weights()
            self.model.get_layer('consent_hl')\
                        .set_weights(consent_weights)
        # # Set learning_rate decay
        # num_train_steps = (labels.shape[0] // self.config.batch_size) \
        #                     * self.config.max_epochs
        # lr_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        #     initial_learning_rate = self.config.learning_rate,
        #     end_learning_rate = 1e-5,
        #     decay_steps = num_train_steps
        # )
        # Compile the model
        assert isinstance(self.config.learning_rate, float), \
            f"Invalid learning_rate ({self.config.learning_rate})"
        self.model.compile(
            loss='categorical_crossentropy', loss_weights=[1, 1],
            optimizer=tf.keras.optimizers.Adam(\
                                    learning_rate=self.config.learning_rate),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                     tfa.metrics.CohenKappa(num_classes=labels.shape[1],
                                            name='kappa'),
                     tfa.metrics.F1Score(num_classes=labels.shape[1],
                                         average='micro')]
        )
        # Split training data into train+val sets
        train_data, val_data = utils.train_val_sampler(
            texts, contexts, labels,
            limit_training_samples = limit_samples,
            batch_size = self.config.batch_size,
            random_state=self.random_state
        )
        # Set callbacks
        callbacks = []
        if self.config.callback_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=self.config.callback_patience,
                                         restore_best_weights=False)
            )
        if self.config.wandb_project:
            self.setup_wandb()
            callbacks.append(
                WandbCallback(monitor="val_consent_output_kappa",
                              mode='max',
                              save_model=False,
                              validation_data=val_data,
                              labels=self.config.codes)
            )
        # Model fit
        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.max_epochs,
            verbose=tf_verbosity,
            callbacks=callbacks
        )
        if save_model:
            Path(os.path.dirname(save_model))\
                .mkdir(parents=True, exist_ok=True)
            self.model.save(save_model)
            with open(os.path.join(save_model, "config.json"), "w+") as f:
                json.dump(self.config.__dict__, f, ensure_ascii=False, indent=4)
        return self


    def setup_wandb(self, config = None):
        if self.config.wandb_project:
            if config is None:
                config = self.config
            self.wandb_run = wandb.init(project=self.config.wandb_project,
                                        config=config.__dict__)


    def finish_wandb(self):
        if self.config.wandb_project and self.wandb_run:
            self.wandb_run.finish()


    def load_cached_context(self, dialog_id):
        """
        Loads cached contextual information of a particular dialog_id.
        """
        if dialog_id in self.cache:
            last_user = self.cache[dialog_id]['last_user']
            previous_codes = self.cache[dialog_id]['previous_codes']
        else: # not found, return False and default_codes
            last_user = False
            previous_codes = deque([self.config.default_code]*self.config.lags)
        return last_user, previous_codes


    def update_cached_context(self, dialog_id, username, current_code_label):
        """
        Updates cached contextual information of a particular dialog_id.
        """
        if dialog_id not in self.cache:
            self.cache[dialog_id] = {
                'last_user': username,
                'previous_codes': \
                    deque( [self.config.default_code]*self.config.lags )
            }
        self.cache[dialog_id]['last_user'] = username
        self.cache[dialog_id]['previous_codes'].appendleft(current_code_label)
        if len(self.cache[dialog_id]['previous_codes'])>self.config.lags:
            self.cache[dialog_id]['previous_codes'].pop()
        return self


    def predict_proba(self,
                      dialog_id: str,
                      username: str,
                      text: str):
        """
        Generates prediction of sent_code and consent_code of one single
        message in probabilities. It loads cached contextual information
        about the referred dialog_id, if it exists. Otherwise, a default
        set of contextual features are used. Cached contextual information
        is updated.

        Args:
            dialog_id (str): Message's dialog_id.
            username (str): Message's username.
            text (str): Message's raw text.

        Returns:
            sent_code (np.array): Probability of each code using only
                sentence embeddings.
            consent_code (np.array): Probability of each code using both
                contextual information and sentence embeddings.
        """
        # load contextual information
        last_user, previous_codes = \
            self.load_cached_context(dialog_id)
        # pack together the context features
        context = np.concatenate([
            [int('?' in text)],
            [int(username == last_user)],
            self.onehot_encoder.transform(
                np.array(previous_codes).reshape(-1,1)
            ).toarray().ravel()
        ]).astype(np.float32).reshape(1,-1)
        # generate model prediction using text and context
        sent_code, consent_code = self.model.predict(
            [tf.constant([text]), tf.constant(context)]
        )
        # if model provide nan outputs (possible when its not well trained)
        #  then use default_code (one hot encoded)
        if np.isnan(sent_code).any():
            sent_code = self.onehot_encode([self.config.default_code])
        if np.isnan(consent_code).any():
            consent_code = self.onehot_encode([self.config.default_code])
        # update contextual information
        self.update_cached_context(dialog_id, username,
            current_code_label=self.onehot_decode(consent_code))
        return sent_code, consent_code


    def onehot_encode(self, labels):
        """
        Transforms string labels into one hot vector representation.
        """
        return self.onehot_encoder.transform(np.array(labels).reshape(-1, 1))\
                                  .toarray()


    def onehot_decode(self, probs: np.array):
        """
        Transforms softmax probabilities (np.array) back into string labels.
        """
        return self.onehot_encoder.inverse_transform(probs)[0][0]


    def predict_sequence(self, dialog_data: List[Message]) -> List[Message]:
        """
        Generates prediction of sent_code and consent_code of one particular
        sequence of messages (dialog_data of one dialog_id).
        To handle inference with multiple dialog_id, use the method
        `predict_proba`.

        Args:
            dialog_data (List[Message], pd.DataFrame): Data of only *one*
                dialog, with at least the attributes `text`, `username`,
                `dialog_id`. If metrics evaluations are being made using
                the output of this function, the `code` attribute should also
                be provided.

        Returns:
            dialog_data (List[Message]): Data with more attributes, referring
                the predicted codes (`sent_code` and `consent_code`).
        """
        assert pd.DataFrame(dialog_data)['dialog_id'].nunique() == 1 , \
            "predict_sequence does only support sequenced predictions on " + \
            "messages of the same dialog_id. If you need to have results " + \
            "from multiple dialog_ids, use this fuction as follows: " + \
            "dialog_data.groupby('dialog_id').apply(consent.predict_sequence)"+\
            ". In this case, we assume type(dialog_data) is a pd.DataFrame."
        # Parse DataFrame as dialog_data
        if type(dialog_data) == pd.DataFrame:
            dialog_data = dialog_data.to_dict(orient="records")

        # Validate dialog_data
        DialogValidator(dialog_data=dialog_data)

        results = list()
        for i, message in enumerate(dialog_data):
            # Generate predictions, append to results
            probas = self.predict_proba(
                dialog_id=message['dialog_id'],
                username=message['username'],
                text=message['text']
            )
            message['sent_code'] = self.onehot_decode(probas[0])
            message['consent_code'] = self.onehot_decode(probas[1])
            results.append(message)
        if type(dialog_data) == pd.DataFrame:
            return pd.DataFrame(results)
        return results


#
