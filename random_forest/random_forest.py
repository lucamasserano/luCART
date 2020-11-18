from __future__ import annotations
from typing import Union, Callable

from luCART.classification_tree import tree

import numpy as np
import pandas as pd
import logging


class RandomForestClassifier(tree.ClassificationTree):

    def __init__(self,
                 n_classifiers,
                 n_covariates: Union[str, int],
                 impurity_function: Union[str, Callable],
                 data_source_type="dataframe",
                 min_datapoints_split=5,
                 debug=False):

        if data_source_type != "dataframe":
            raise ValueError(f"Currently only Pandas DataFrames are supported, got {data_source_type}")

        # TODO: which parameters from mother class are actually useful?
        super().__init__(impurity_function=impurity_function,
                         min_datapoints_split=min_datapoints_split,
                         data_source_type=data_source_type,
                         debug=debug)

        self.n_classifiers = n_classifiers
        self.n_covariates = n_covariates

        self.forest = dict()

    def choose_split(self, impurity_function: Callable, decision_path: list) -> dict:

        if self.debug:
            np.random.seed(7)

        if isinstance(self.n_covariates, int):
            covariates = np.random.choice(self.covariates, size=self.n_covariates, replace=False)
        elif self.n_covariates == "sqrt":
            n_covariates = int(round(np.sqrt(len(self.covariates))))
            covariates = np.random.choice(self.covariates, size=n_covariates, replace=False)
        else:
            # TODO: implement other ways to define n_features
            covariates = None
            pass

        logging.debug(f"Choosing split with covariates set {covariates}")

        # TODO: can parallelize this?
        choose_best_split_covariate = [self.covariate_impurity_reduction(impurity_function=impurity_function,
                                                                         covariate_name=covariate,
                                                                         decision_path=decision_path)
                                       for covariate in covariates]

        max_reduction_index = np.argmax([result[1] for result in choose_best_split_covariate])

        return {"best_covariate": covariates[max_reduction_index],
                "best_split_point": choose_best_split_covariate[max_reduction_index][0]}

    def fit(self, data: pd.Dataframe, label: str) -> None:

        self.label = label
        self.covariates = self.data_handler.get_covariate_names(label=label, data=data)

        for iteration in range(self.n_classifiers):
            logging.info(f"Building tree number {iteration + 1}")

            if self.debug:
                np.random.seed(7)
            bootstrap_indices = np.random.choice(data.index, size=len(data.index), replace=True)
            bootstrap_sample = data.loc[bootstrap_indices, :]

            # temporarily set default data to bootstrap_sample and use that for building trees
            self.data = bootstrap_sample

            self.build_tree(impurity_function=self.impurity_function, decision_path=None)
            self.forest[f"tree_{iteration + 1}"] = {"root": self.root, "leaves": self.leaves,
                                                    "tree_cardinality": self.tree_cardinality,
                                                    "keys": self.keys}
            self.root = None
            self.leaves = dict()
            self.tree_cardinality = 0
            self.keys = set()

        # reset default data to entire dataset when done fitting
        self.data = data

    def rf_predict(self,
                   new_data: pd.DataFrame,
                   training_data=None,
                   prediction_threshold=0.5) -> Union[np.ndarray, pd.DataFrame]:

        # TODO: call this method predict() and use super().predict below

        logging.info(f"Predicting using random forest for dataset with dimensions {new_data.shape}")

        if training_data is None:
            training_data = self.data

        # TODO: implement equivalent of return_new_data=True
        trees_prediction = [pd.DataFrame(self.predict(new_data=new_data,
                                                      return_new_data=False,
                                                      training_data=training_data,
                                                      root=self.forest[tree_name]["root"],
                                                      leaves=self.forest[tree_name]["leaves"],
                                                      tree_keys=self.forest[tree_name]["keys"]))
                            for tree_name in self.forest]

        aggregate_predictions = pd.concat(trees_prediction, axis=1, ignore_index=True)
        aggregate_predictions["mean"] = aggregate_predictions.mean(axis=1)
        aggregate_predictions["final_prediction"] = 0
        aggregate_predictions.loc[aggregate_predictions["mean"] > prediction_threshold, "final_prediction"] = 1

        return aggregate_predictions.loc[:, "final_prediction"].to_numpy()
