from __future__ import annotations
from typing import Union, Callable
import numbers

import numpy as np
import pandas as pd
from itertools import repeat
import warnings
import logging

from luCART.classification_tree import impurity
from luCART.data_handling import data_handler


class Node:

    def __init__(self,
                 parent_split_covariate, parent_split_point, parent_split_relation):

        # TODO: split_covariate should be checked to be in columns

        Node.check_relation(parent_split_relation)

        if parent_split_relation == "root":
            self.key = "root"
            self.value = "root"
            self.parent_split_covariate = "root"
            self.parent_split_point = "root"
            self.parent_split_relation = "root"
        else:
            self.key = f"{parent_split_covariate} {parent_split_relation} {parent_split_point}"
            if isinstance(parent_split_point, str):
                parent_split_point = f"'{parent_split_point}'"
            self.value = f"(<df>['{parent_split_covariate}'] {parent_split_relation} {parent_split_point})"
            self.parent_split_covariate = parent_split_covariate
            self.parent_split_point = parent_split_point
            self.parent_split_relation = parent_split_relation

        self.split_covariate = None
        self.split_point = None

        self.left = None
        self.right = None
        self.parent = None

        self.children_keys = dict()

    @staticmethod
    def check_relation(split_relation):
        # TODO: need != as well?
        valid = ["<", ">=", "==", "root"]
        if split_relation not in valid:
            raise ValueError(f"Split_relation must be one of {valid}.")

    # TODO: define setters and deleters if necessary


class ClassificationTree:

    def __init__(self,
                 impurity_function: Union[str, Callable],
                 data_source_type: str,
                 min_datapoints_split=5,
                 alpha=None,
                 n_folds=5,
                 stratified_folds=False,
                 debug=False):

        # TODO: define assert statements to check if parameters are valid

        self.debug = debug

        # very basic logging, for the moment used only for debugging purposes
        if self.debug:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO
        logging.basicConfig(
            filename="./luCART/debugging.log",
            level=logging_level,
            format='%(asctime)s %(levelname)s %(message)s')

        # TODO: provide other ways to stop splitting?
        # leaves with n_obs <= min_datapoints_split stop splitting procedure
        self.min_datapoints_split = min_datapoints_split
        self.alpha = alpha  # regularization parameter
        self.alpha_cv = None

        self.n_folds = n_folds
        self.stratified_folds = stratified_folds

        self.root = None
        self.leaves = dict()
        self.tree_cardinality = 0
        self.keys = set()

        self.covariates = None
        self.label = None

        self.data_source_type = data_source_type
        if data_source_type == "SQL":
            self.data_handler = data_handler.SQLHandler(impurity_function=impurity_function)
        elif data_source_type == "dataframe":
            self.data_handler = data_handler.DataframeHandler(impurity_function=impurity_function, data=None)
        else:
            raise ValueError(f"Only SQL and Pandas DataFrames are currently supported, got '{data_source_type}'")

    @property
    def impurity_function(self):
        return self.data_handler.impurity_function

    @property
    def data(self):
        return self.data_handler.data

    @data.setter
    def data(self, data):
        self.data_handler.data = data

    def update_children_keys(self, target_keys: Union[list, str], current_node: Node, mode: str) -> None:
        # climbs tree from parent to parent to update each children key when a new node is created

        if isinstance(target_keys, str):
            target_keys = [target_keys]

        mode_values = ["a", "d"]
        if mode not in mode_values:
            raise ValueError(f"mode must be among {mode_values}, got {mode}")

        if current_node.key == "root":
            return
        # if coming from left then target_key is under left branch
        # TODO: could not simply use parent.left.key or parent.right.key?
        elif current_node.parent.children_keys[current_node.key] == "left":
            if mode == "a":
                for target_key in target_keys:
                    current_node.parent.children_keys[target_key] = "left"
            else:
                for target_key in target_keys:
                    current_node.parent.children_keys.pop(target_key)
        else:
            if mode == "a":
                for target_key in target_keys:
                    current_node.parent.children_keys[target_key] = "right"
            else:
                for target_key in target_keys:
                    current_node.parent.children_keys.pop(target_key)

        self.update_children_keys(target_keys=target_keys, current_node=current_node.parent, mode=mode)

    def add_node(self, parent_node: Union[Node, None], new_node: Node, where: Union[str, None]) -> None:
        # make a parent Node point to a newly created Node and update all relevant attributes

        if parent_node is None:
            self.keys.add(new_node.key)
            self.root = new_node
            self.leaves[new_node.key] = new_node
            self.tree_cardinality += 1
        else:
            if where == "left":
                parent_node.left = new_node
                new_node.parent = parent_node
                # only update direct parent, other ancestors are updated by dedicated method
                parent_node.children_keys[new_node.key] = "left"

            elif where == "right":
                parent_node.right = new_node
                new_node.parent = parent_node
                parent_node.children_keys[new_node.key] = "right"

            else:
                raise ValueError(f"Parameter 'where' expected either 'left' or 'right', '{where}' was given")

            if new_node.key in self.keys:
                warnings.warn(f"{new_node.key} already existing in tree.\
                              \nCan cause unexpected behaviour when searching for {new_node.key}")
            self.keys.add(new_node.key)
            # if parent node has not already been deleted from leaves given a previous child addition
            if parent_node.key in self.leaves:
                self.leaves.pop(parent_node.key)
            self.leaves[new_node.key] = new_node
            self.tree_cardinality += 1
            self.update_children_keys(target_keys=new_node.key, current_node=parent_node, mode="a")

    def search_node(self, target_node_key: str, current_node=None) -> Node:
        # raise KeyError if node not in tree, else return Node
        if target_node_key not in self.keys:
            raise KeyError(f"Target node key {target_node_key} not in tree")
        else:
            if current_node is None:
                current_node = self.root

            if current_node.key == target_node_key:
                return current_node
            # target_node key (split_covariate <>== split_point) is in left branch
            # TODO: what if key is both in left and right branch? Is this possible?
            elif current_node.children_keys[target_node_key] == "left":
                return self.search_node(target_node_key=target_node_key,
                                        current_node=current_node.left)
            else:  # right branch
                return self.search_node(target_node_key=target_node_key,
                                        current_node=current_node.right)

    def remove_node(self, node) -> None:
        # is this needed?
        pass

    def prune_at_node(self, subtree_root_key: str) -> None:

        logging.info(f"Pruning subtree rooted at {subtree_root_key}")

        subtree_root = self.search_node(target_node_key=subtree_root_key)
        children_keys = list(subtree_root.children_keys.copy().keys())

        subtree_root.children_keys = dict()
        subtree_root.left = None
        subtree_root.right = None
        subtree_root.split_covariate = None
        subtree_root.split_point = None

        self.leaves[subtree_root.key] = subtree_root
        for key in children_keys:
            subtree_root.parent.children_keys.pop(key)
            if key in self.leaves:
                self.leaves.pop(key)
            self.keys.remove(key)
        self.tree_cardinality -= len(children_keys)

        self.update_children_keys(target_keys=children_keys, current_node=subtree_root.parent, mode="d")

    def decision_path(self, target_node: Node, current_node=None, tree_keys=None):
        # traverse the tree and recursively create an iterable of multiple booleans (one for each
        # splitting condition). Data should not be touched until the desired node is reached to avoid using memory.
        # At the end Pandas masking functionality can be used to obtain the relevant portion of data, but this should
        # be done only by get_data.

        # TODO: should return list of node keys if using SQLHandler? Node value is based on Pandas masking

        if tree_keys is None:
            tree_keys = self.keys
        if current_node is None:
            current_node = self.root

        if target_node.key not in tree_keys:
            raise KeyError(f"Target node key {target_node.key} not in tree")

        if current_node.key == target_node.key:
            if current_node.key == "root":
                return []  # root node of the tree contains all the data -> no boolean mask
            else:
                return [target_node.value]

        # target_node key (split_covariate <>== split_point) is in left branch
        # TODO: what if key is both in left and right branch? Is this possible?
        if current_node.children_keys[target_node.key] == "left":
            if current_node.key == "root":
                return self.decision_path(target_node=target_node,
                                          current_node=current_node.left,
                                          tree_keys=tree_keys)
            else:
                return [current_node.value] + self.decision_path(target_node=target_node,
                                                                 current_node=current_node.left,
                                                                 tree_keys=tree_keys)
        else:  # right branch
            if current_node.key == "root":
                return self.decision_path(target_node=target_node,
                                          current_node=current_node.right,
                                          tree_keys=tree_keys)
            else:
                return [current_node.value] + self.decision_path(target_node=target_node,
                                                                 current_node=current_node.right,
                                                                 tree_keys=tree_keys)

    def get_data(self, target_node=None, data=None, columns=None, decision_path=None) -> pd.DataFrame:
        # explores self.tree and returns data that falls into the specified target_node.

        if data is None:
            data = self.data
        if columns is None:
            columns = data.columns

        # TODO: either target_node or decision_path must not be None

        # retrieve decision path and create the boolean mask to use
        if decision_path is None:
            decision_path = self.decision_path(target_node=target_node, current_node=None)

        return self.data_handler.get_data(decision_path=decision_path, columns=columns, data=data)

    def is_valid_tree(self, root=None, root_decision_path=None):
        # TODO: adapt for SQL integration
        # TODO: check other things if they come to mind
        # TODO: put each single check in separate functions?
        if root is None:
            root = self.root

        # check that each node has data under it
        if root_decision_path is None:
            root_decision_path = self.decision_path(target_node=root)
        root_data = self.get_data(target_node=root, data=self.data, decision_path=root_decision_path)
        assert not root_data.empty, f"Dataset under {root.key} is empty"

        # check split rule is respected
        left_decision_path = root_decision_path.append(root.left.value)
        right_decision_path = root_decision_path.append(root.right.value)
        left_data = self.get_data(target_node=root.left, data=self.data, decision_path=left_decision_path)
        right_data = self.get_data(target_node=root.right, data=self.data, decision_path=right_decision_path)
        left_split_mask = root.left.value.replace("<df>", "left_data")
        right_split_mask = root.right.value.replace("<df>", "right_data")
        assert left_data.loc[eval(right_split_mask), :].empty(), \
            f"Node with {left_split_mask} should not have data with {right_split_mask}"
        assert right_data.loc[eval(left_split_mask), :].empty(), \
            f"Node with {right_split_mask} should not have data with {left_split_mask}"

        # check that combined_leaves_data == root_data
        leaves_data = list(map(self.get_data, self.leaves.values(), repeat(self.data)))
        combined_leaves_data = pd.concat(leaves_data, axis=0, join="inner", ignore_index=True) \
            .sort_values(by=self.data.columns[0])
        root_data = self.get_data(target_node=self.root, data=self.data).sort_values(by=self.data.columns[0])
        assert combined_leaves_data.reset_index(drop=True).equals(root_data)

        return True

    def covariate_impurity_reduction(self,
                                     impurity_function: Callable,
                                     covariate_name: str,
                                     decision_path: list) -> tuple:

        logging.info(f"Computing covariate impurity reduction for {covariate_name}")

        split_points = self.data_handler.find_split_points(covariate_name=covariate_name, decision_path=decision_path)

        # compute observation counts and p(y=1|region) for left, right and full regions, for each split point
        impurity_reduction_constructors = \
            self.data_handler.impurity_reduction_constructors(split_points=split_points,
                                                              covariate_name=covariate_name,
                                                              label=self.label,
                                                              decision_path=decision_path)

        # compute impurity reductions for each split point and get maximum
        impurity_reductions = impurity.impurity_reduction(impurity_function=impurity_function,
                                                          p=impurity_reduction_constructors[0],
                                                          n_obs_left=impurity_reduction_constructors[1],
                                                          n_obs_right=impurity_reduction_constructors[2],
                                                          p_left=impurity_reduction_constructors[3],
                                                          p_right=impurity_reduction_constructors[4],
                                                          )
        max_reduction_index = np.argmax(impurity_reductions)

        logging.debug(f"{covariate_name} best split point: {split_points[max_reduction_index]}.\
                      Impurity reduction: {impurity_reductions[max_reduction_index]}")

        return split_points[max_reduction_index], impurity_reductions[max_reduction_index]

    def choose_split(self, impurity_function: Callable, decision_path: list) -> dict:

        logging.info("Choosing best split")

        # TODO: can parallelize this?
        choose_best_split_covariate = [self.covariate_impurity_reduction(impurity_function=impurity_function,
                                                                         covariate_name=covariate,
                                                                         decision_path=decision_path)
                                       for covariate in self.covariates]

        max_reduction_index = np.argmax([result[1] for result in choose_best_split_covariate])
        logging.debug(f"Chose {self.covariates[max_reduction_index]} and \
                        {choose_best_split_covariate[max_reduction_index][0]}")

        return {"best_covariate": self.covariates[max_reduction_index],
                "best_split_point": choose_best_split_covariate[max_reduction_index][0]}

    def build_tree(self, impurity_function: Callable, decision_path=None, current_node=None) -> None:

        label = self.label

        if current_node is None:
            logging.info("Adding root node")
            current_node = Node(parent_split_covariate="root", parent_split_point="root", parent_split_relation="root")
            self.add_node(parent_node=None, new_node=current_node, where=None)
            decision_path = []

        # stop if all data points in the node are of the same class
        if self.data_handler.check_if_pure(label=label, decision_path=decision_path):
            logging.info(f"Stopped at {current_node.key}: all data points are of the same class")
            self.leaves[current_node.key] = current_node
            return
        # TODO: could be better to check this while splitting parent to avoid too few observations?
        # stop if data under current_node is already small enough
        elif self.data_handler.check_if_small(min_obs_split=self.min_datapoints_split, decision_path=decision_path):
            logging.info(f"Stopped at {current_node.key}: data under current_node is already small enough")
            self.leaves[current_node.key] = current_node
            return
        else:
            best_split = self.choose_split(impurity_function=impurity_function, decision_path=decision_path)
            # TODO: it would be more appropriate if these were inside add_node method
            current_node.split_covariate = best_split["best_covariate"]
            current_node.split_point = best_split["best_split_point"]

            # TODO: only valid for numeric predictors. Write to handle categorical predictors too
            left_node = Node(parent_split_covariate=current_node.split_covariate,
                             parent_split_point=current_node.split_point,
                             parent_split_relation="<")
            right_node = Node(parent_split_covariate=current_node.split_covariate,
                              parent_split_point=current_node.split_point,
                              parent_split_relation=">=")

            # add left and right nodes and update all relevant attributes
            self.add_node(parent_node=current_node, new_node=left_node, where="left")
            self.add_node(parent_node=current_node, new_node=right_node, where="right")

            # recursively build tree on new left and right nodes
            left_decision_path = decision_path + [current_node.left.value]
            right_decision_path = decision_path + [current_node.right.value]
            self.build_tree(current_node=current_node.left, impurity_function=impurity_function,
                            decision_path=left_decision_path)
            self.build_tree(current_node=current_node.right, impurity_function=impurity_function,
                            decision_path=right_decision_path)

    def prune_node_cost_increase(self, node: Node, decision_path: list) -> float:
        # computes relative misclassification cost of pruning the tree at node (g(t) in notes)

        logging.debug(f"Computing pruning cost increase for subtree rooted at {node.key}")

        node_misclassification_cost = \
            self.data_handler.node_misclassification_cost(node_key=node.key,
                                                          decision_path=decision_path,
                                                          label=self.label)
        node_leaves_set = set(node.children_keys).intersection(set(self.leaves))
        node_leaves_misclassification_cost = np.sum(
            [self.data_handler.node_misclassification_cost(
                node_key=self.leaves[leaf].key,
                decision_path=self.decision_path(target_node=self.leaves[leaf]),
                label=self.label
            )
             for leaf in node_leaves_set]
        )

        logging.debug(f"node_cost: {node_misclassification_cost}, sub_tree_leaves: {node_leaves_set}, \
                        leaves_cost: {node_leaves_misclassification_cost}")
        return (node_misclassification_cost - node_leaves_misclassification_cost)/(len(node_leaves_set)-1)

    def pruning_step(self, node: Node, decision_path=None):

        if node.key == "root":
            return self.pruning_step(node=node.left, decision_path=[]) + \
                self.pruning_step(node=node.right, decision_path=[])
        elif node.key not in self.leaves:
            decision_path += [node.value]
            return [(node.key, self.prune_node_cost_increase(node=node, decision_path=decision_path))] + \
                self.pruning_step(node=node.left, decision_path=decision_path) + \
                self.pruning_step(node=node.right, decision_path=decision_path)
        else:
            # we are interested in the least cost increase. In this way leaves are ignored. None is a placeholder
            return [(None, np.inf)]

    def prune_tree(self, alpha: numbers.Number) -> None:

        logging.info(f"Pruning tree with alpha = {alpha}")

        optimally_pruned = False
        while not optimally_pruned:
            nodes_pruning_cost_increases = self.pruning_step(self.root)
            pruning_step_best_node_key, alpha_star = min(nodes_pruning_cost_increases, key=lambda elem: elem[1])
            if alpha_star <= alpha:
                if alpha_star < -1e-5:
                    msg = f"alpha* = {alpha_star}. Is this still simply underflow?"
                    warnings.warn(msg)
                    logging.warning(msg)

                logging.info(f"Computed alpha* = {alpha_star}")
                self.prune_at_node(subtree_root_key=pruning_step_best_node_key)
            else:
                logging.info(f"alpha* = {alpha_star}. Tree is optimally pruned")
                optimally_pruned = True

    def fit(self, data: Union[str, pd.Dataframe], label: str) -> None:
        # this should be a wrapper around build_tree (and prune_tree if requested)

        self.data = data
        self.label = label
        self.covariates = self.data_handler.get_covariate_names(label=label, data=data)

        if (self.alpha is None) or (self.alpha == 0):  # no pruning
            logging.info("Building tree without pruning")
            self.build_tree(impurity_function=self.impurity_function, decision_path=None)
            return
        elif isinstance(self.alpha, numbers.Number):
            logging.info("Building and pruning tree")
            self.build_tree(impurity_function=self.impurity_function, decision_path=None)
            self.prune_tree(alpha=self.alpha)
            return
        elif isinstance(self.alpha, (list, np.ndarray)):  # cv with given range of alpha
            alpha = self.choose_alpha(n_folds=self.n_folds, stratified_folds=self.stratified_folds,
                                      data=data, alpha_range=self.alpha)
            self.alpha_cv = alpha
            logging.info("Building and pruning tree after cv")
            self.build_tree(impurity_function=self.impurity_function, decision_path=None)
            self.prune_tree(alpha=alpha)
            return
        elif self.alpha == "cv":  # cv with default range of alpha
            alpha = self.choose_alpha(n_folds=self.n_folds,
                                      stratified_folds=self.stratified_folds, data=data)
            self.alpha_cv = alpha
            logging.info("Building and pruning tree after cv")
            self.build_tree(impurity_function=self.impurity_function, decision_path=None)
            self.prune_tree(alpha=alpha)
            return

    def predict(self,
                new_data: pd.DataFrame,
                return_new_data=False,
                training_data=None,
                root=None,
                leaves=None,
                tree_keys=None) -> Union[np.ndarray, pd.DataFrame]:

        # TODO: is there a simpler way to obtain predictions?

        logging.info(f"Predicting for dataset with dimensions {new_data.shape}")

        if training_data is None:
            training_data = self.data
        if root is None:
            root = self.root
        if leaves is None:
            leaves = self.leaves
        if tree_keys is None:
            tree_keys = self.keys

        # check new_data is aligned with training data
        new_data_columns = self.data_handler.get_covariate_names(label=self.label, data=new_data)
        # TODO: what if columns not in same order?
        if not np.all(new_data_columns == self.covariates):
            logging.error(f"Mismatch between columns in training and columns in new_data. Aborting")
            raise KeyError(f"Columns in new_data must be the same as in training data")

        # match leaf with new_data and training data falling under it
        # TODO: do this block in a single for loop
        leaves_decision_paths = [self.decision_path(target_node=leaves[leaf], current_node=root, tree_keys=tree_keys)
                                 for leaf in leaves]
        data_leaves = [(self.get_data(data=new_data,
                                      decision_path=decision_path),
                        self.get_data(data=training_data,
                                      columns=[self.label],
                                      decision_path=decision_path)
                        )
                       for decision_path in leaves_decision_paths]

        return self.data_handler.predict_majority_class(data_leaves=data_leaves,
                                                        label=self.label,
                                                        return_new_data=return_new_data)

    def choose_alpha(self,
                     n_folds: int,
                     stratified_folds: bool,
                     data: pd.DataFrame,
                     alpha_range=None) -> float:
        # apply cross validation to choose the best alpha

        if data is None:
            data = self.data

        if alpha_range is None:
            # TODO: what are reasonable values of alpha?
            alpha_range = np.linspace(0, 1, 20)

        logging.info(f"Applying cross validation to choose alpha among {alpha_range}")

        if stratified_folds:
            # TODO: needed if classes are unbalanced
            folds_indices = None
            pass
        else:
            if self.debug:
                np.random.seed(7)
            shuffled_index = np.random.permutation(data.index.to_numpy())
            folds_indices = np.array_split(shuffled_index, n_folds)

        generalization_errors = []
        for alpha in alpha_range:
            logging.info(f"cv with alpha = {alpha}")
            fold_errors = []
            for fold_number, fold in enumerate(folds_indices):
                logging.info(f"cv fold {fold_number + 1}")
                temp_classification_tree = ClassificationTree(impurity_function=self.impurity_function,
                                                              data_source_type=self.data_source_type,
                                                              min_datapoints_split=self.min_datapoints_split,
                                                              alpha=alpha)
                training_data = data.loc[~data.index.isin(fold), :]
                test_data_x = data.loc[fold, :].drop(labels=self.label, axis=1)
                test_data_y = data.loc[fold, self.label].to_numpy()
                temp_classification_tree.fit(data=training_data, label=self.label)
                predictions = temp_classification_tree.predict(new_data=test_data_x,
                                                               return_new_data=False,
                                                               training_data=training_data)
                # TODO: only valid for binary classification (?)
                fold_generalization_error = np.sum(predictions != test_data_y)/len(predictions)
                fold_errors.append(fold_generalization_error)

            average_folds_error = np.mean(fold_errors)
            logging.info(f"alpha = {alpha}, cv_error = {average_folds_error}")
            generalization_errors.append(average_folds_error)

        return alpha_range[np.argmin(generalization_errors)]

    def show(self, root: Node, level=0):

        print(" " * level, root.key, sep="")

        if root.left is not None:
            print(" " * level, "left:", sep="")
            self.show(root.left, level + 2)

        if root.right is not None:
            print(" " * level, "right:", sep="")
            self.show(root.right, level + 2)
