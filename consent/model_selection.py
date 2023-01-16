import os
from time import time
from typing import Union, List
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterSampler, GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import fire

from consent import ConSent, Config, HYPERPARAMETERS
import consent.utils as utils



class Handler:
    """
    Command-line interface for preparing & analysing ConSent models.
    """

    def __init__(self,
                 data_file_path: Union[str, List],
                 config_file_path: str = None,
                 load_model: str = None,
                 random_state: int = 0):
        assert config_file_path is not None or load_model, \
            "Either provide a `config_file_path` or `load_model`."
        # load config
        if config_file_path is not None:
            with open(config_file_path, 'r') as f:
                self.config = Config(**json.load(f))
            if load_model:
                with open(os.path.join(load_model, 'config.json'), 'r') as f:
                    load_config = Config(**json.load(f))
                # append new dataset_name to config.datasets
                self.config.datasets.append(self.config.dataset_name)
        else:
            with open(os.path.join(load_model, 'config.json'), 'r') as f:
                self.config = Config(**json.load(f))
        if load_model is not None:
            self.config.load_weights = load_model

        # load data
        if isinstance(data_file_path, list):
            df = []
            for file_path in data_file_path:
                df.append(utils.load_data(file_path,
                                    code_name=self.config.code_name))
            self.data_df = pd.concat(df)
        else:
            self.data_df = utils.load_data(data_file_path,
                                     code_name=self.config.code_name)
        self.random_state = random_state


    def save_results(self,
                     data,
                     output_file_path,
                     remove_keys=None):
        """
        Saves data to csv.
        """
        Path(os.path.dirname(output_file_path))\
            .mkdir(parents=True, exist_ok=True)
        if type(remove_keys) == list:
            data = data.drop(columns=remove_keys)
        data.to_csv(output_file_path, encoding='utf-8')


    def compute_metrics(self, y_true, y_pred, config):
        """
        Computes all the evaluation metrics using the results of
        ConSent.predict_sequence(). `config` needs to be assigned such
        that results are attributed to a given configuration.
        """
        metrics = {
            'accuracy': accuracy_score(
                y_true, y_pred
            ),
            'kappa': cohen_kappa_score(
                y_true, y_pred
            ),
            'micro-f1': f1_score(
                y_true, y_pred,
                average='micro', zero_division=0
            ),
            'classification_report': classification_report(
                y_true, y_pred,
                output_dict=True, zero_division=0
            ),
            'confusion_matrix': multilabel_confusion_matrix(
                y_true, y_pred
            ),
            'config': json.dumps(config.__dict__,
                                 ensure_ascii=False,
                                 default=utils.convert)
        }
        return metrics


    def train(self,
              save_model: str = None):
        """
        Trains a model using the configuration file provided.
        """
        print(self.config)

        consent = ConSent(self.config, load=False,
                          random_state=self.random_state)
        consent.setup_wandb()
        consent.train(self.data_df, tf_verbosity=2, save_model=save_model)
        consent.finish_wandb()


    def test(self,
             dataset_name: str,
             output_file_path: str = None):
        """
        Docs
        """
        assert dataset_name is not None, \
            "To `test`, you need to provide `dataset_name`."
        assert isinstance(self.config.load_weights, str), \
            "Provide load_model (a path to a trained model)."
        self.config.caller = 'test'
        self.config.dataset_name = dataset_name
        consent = ConSent(self.config,
                          load=self.config.load_weights,
                          random_state=self.random_state)
        consent.setup_wandb()
        predictions = self.data_df.groupby('dialog_id')\
                                  .apply(consent.predict_sequence)
        results = pd.concat(predictions.apply(pd.DataFrame).values)\
                    .reset_index()
        metrics = self.compute_metrics(results['code'],
                                       results['consent_code'],
                                       self.config)
        consent.wandb_run.log(metrics)
        consent.finish_wandb()
        if output_file_path is not None:
            self.save_results(pd.Series(metrics),
                              output_file_path=output_file_path)


    def hyperparameter_tuning(self,
                              val_size: float = 0.2,
                              n_iter: int = -1,
                              output_file_path: str = None):
        """
        Finds the performance of training models with different parameters.
        """
        self.config.caller = 'hyperparameter_tuning'
        t0 = time()
        experiment_config = self.config.__dict__
        experiment_grid = {
            key: experiment_config[key] for key in experiment_config \
            if isinstance(experiment_config[key], list) and \
               key in HYPERPARAMETERS
        }
        # split train-validation sets
        train_data, test_data = \
            utils.train_test_split(self.data_df,
                                   test_size=val_size,
                                   random_state=self.random_state)
        # find max n_iter if user provides -1
        if n_iter == -1:
            n_iter = 1
            for k in experiment_grid.keys():
                n_iter *= len(experiment_grid[k])
        print(f"Hyperparameter tuning :: n_iter={n_iter}\n{experiment_grid}")
        # sampling parameter combinations
        parameter_sampler = ParameterSampler(
            param_distributions=experiment_grid,
            n_iter=n_iter, random_state=self.random_state
        )
        # start looping on n_iter parameter combinations
        metrics = list()
        for i, parameters in enumerate(parameter_sampler):
            # overwrite experiment config with the parameters sampled
            config = Config(**{**self.config.__dict__, **parameters})
            print(config)
            # train a ConSent model from scratch
            consent = ConSent(config, load=False,
                              random_state=self.random_state)
            consent.setup_wandb(config)
            consent.train(train_data, tf_verbosity=2)
            # generate predictions
            predictions = test_data.groupby('dialog_id')\
                                           .apply(consent.predict_sequence)
            results = pd.concat(predictions.apply(pd.DataFrame).values)\
                        .reset_index()
            consent.cache = {} # clear cache for the next model
            # Evaluation metrics
            m = self.compute_metrics(results['code'],
                                     results['consent_code'],
                                     config)
            metrics.append(m)
            consent.wandb_run.log(m)
            consent.finish_wandb()
        # Save results
        metrics = pd.DataFrame(metrics)
        if output_file_path is not None:
            self.save_results(metrics, output_file_path=output_file_path)
        t1 = round(time()-t0)
        print("> ConSent hyperparameter_tuning:")
        print(f">> Finished n_iter={n_iter} in {t1}s.")


    def evaluation(self, test_data, consent, return_sent_metrics=False):
        predictions = test_data.groupby('dialog_id')\
                               .apply(consent.predict_sequence)
        results = pd.concat(predictions.apply(pd.DataFrame).values)\
                    .reset_index()
        # evaluation metrics
        sent_m = self.compute_metrics(results['code'],
                                      results['sent_code'],
                                      self.config)
        consent_m = self.compute_metrics(results['code'],
                                 results['consent_code'],
                                 self.config)
        if return_sent_metrics:
            return sent_m, consent_m
        else:
            return consent_m


    def cross_validation(self,
                         folds: int = 5,
                         pct_dialog_train: int = -1,
                         output_file_path: str = None,
                         **kwargs):
        """
        Finds the performance of training models cross-validation.
        This is useful to get a more precise and reliable estimation of the
        performance of a model given the config provided.
        """
        self.config.caller = kwargs.get("caller", "cross_validation")
        # Get cv splits from all the data
        cv = GroupKFold(n_splits=folds)
        splits = cv.split(y=self.data_df['code'],
                          X=np.zeros(self.data_df.shape[0]),
                          groups=self.data_df['dialog_id'])
        # Iterate training with each cv split
        sent_metrics, consent_metrics = list(), list()
        for fold, (train_index, test_index) in enumerate(splits):
            train_data = self.data_df.iloc[train_index]
            if 0 < pct_dialog_train <= 1:
                self.config.n_samples = pct_dialog_train
                rng = np.random.default_rng()
                n_groups = int(train_data['dialog_id'].nunique() \
                               * pct_dialog_train)
                grps = rng.choice(train_data['dialog_id'], n_groups,
                                  replace=False, shuffle=False)
                train_data = train_data.loc[train_data['dialog_id'].isin(grps)]
            test_data = self.data_df.iloc[test_index]

            # train from scratch
            consent = ConSent(self.config, load=False,
                              random_state=self.random_state)
            consent.setup_wandb()
            consent.train(train_data, tf_verbosity=2)
            # generate predictions
            sent_m, consent_m = self.evaluation(test_data, consent,
                                                return_sent_metrics=True)
            sent_metrics.append(sent_m)
            consent_metrics.append(consent_m)
            consent.wandb_run.log(consent_m)
            consent.finish_wandb()
        # Join everything
        sent_metrics = pd.DataFrame(sent_metrics)
        sent_metrics.columns = sent_metrics.columns + "_sent"
        consent_metrics = pd.DataFrame(consent_metrics)
        consent_metrics.columns = consent_metrics.columns + "_consent"
        metrics = pd.concat([consent_metrics, sent_metrics], axis=1)\
                    .reset_index()
        metrics.index.name = 'fold'

        # Save results
        if output_file_path is not None:
            self.save_results(metrics,
                              output_file_path=output_file_path)
        return metrics


    def learning_curves(self,
                        test_data_file_path: str,
                        n_sizes: int = 5,
                        folds: int = 5,
                        output_file_path: str = None):
        """
        Determines cross-validated training and test scores given different
        sample sizes.
        """
        self.config.caller = 'learning_curves'
        # load test data
        test_data = utils.load_data(test_data_file_path,
                              code_name=self.config.code_name)
        n_samples = np.linspace(0, self.data_df.shape[0], n_sizes+1)\
                      .astype(int)[1:] # skip the first 0
        metrics = []
        sample_sizes = []
        trial = 0
        for n in n_samples:
            metrics.append(
                self.cross_validation(folds=folds,
                                      pct_dialog_train=n/self.data_df.shape[0],
                                      caller='learning_curves')
            )
            for i in range(folds):
                sample_sizes.append(n)

        metrics = pd.concat(metrics, axis=0)\
                    .reset_index()
        metrics['sample_size'] = sample_sizes
        # Save results
        if output_file_path is not None:
            self.save_results(metrics,
                              output_file_path=output_file_path)
        return


    def fine_tune_experiment(self,
                             test_data_file_path: str,
                             dataset_name: str,
                             output_file_path: str = None):
        """
        Docs
        """
        self.config.caller = 'fine_tune_experiment'
        self.config.dataset_name = dataset_name
        # load test data
        test_data = utils.load_data(test_data_file_path,
                              code_name=self.config.code_name)
        consent_metrics = []
        learning_rates = []

        for lr in [1e-3, 5e-4, 1e-4, 5e-5]:
            self.config.learning_rate = lr
            consent = ConSent(self.config,
                              random_state=self.random_state)
            consent.setup_wandb()
            consent.train(self.data_df,
                          tf_verbosity=2)
            m = self.evaluation(test_data, consent)
            consent_metrics.append(m)
            consent.wandb_run.log(m)
            learning_rates.append(lr)
            consent.finish_wandb()

        # Save results
        consent_metrics = pd.DataFrame(consent_metrics)
        consent_metrics.columns = consent_metrics.columns + "_consent"
        # Save results
        if output_file_path is not None:
            self.save_results(consent_metrics,
                              output_file_path=output_file_path)


    def fine_tune(self,
                  save_model: str):
        """
        Docs
        """
        self.config.caller = 'fine_tune'
        self.train(save_model=save_model)
        return



if __name__=='__main__':
    fire.Fire(Handler)
