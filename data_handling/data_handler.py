# A module to ensure all data is (if needed) read/prepared/written/used consistently.
from typing import Union, Callable
import pandas as pd
import numpy as np
import logging
import psycopg2
import getpass

from luCART.classification_tree import impurity


# TODO: ensure all columns do not have spaces in name
def column_name_check():
    pass


# TODO: ensure response is encoded as 0-1
def check_encoding():
    pass


class DataframeHandler:

    def __init__(self,
                 impurity_function: Union[str, Callable],
                 data: Union[None, pd.DataFrame]):

        if impurity_function == "gini":
            self.impurity_function = impurity.gini_index
        elif impurity_function == "bayes":
            self.impurity_function = impurity.bayes_error
        elif impurity_function == "cross_entropy":
            self.impurity_function = impurity.cross_entropy
        elif isinstance(impurity_function, Callable):
            self.impurity_function = impurity_function
        else:
            raise ValueError(f"Impurity function not supported, got '{impurity_function}'")

        self.data = data

    def get_covariate_names(self,
                            label: str,
                            data=None):

        if data is None:
            data = self.data

        if label not in data.columns:
            return data.columns
        else:
            return data.drop(labels=label, axis=1).columns

    def get_data(self,
                 decision_path: list,
                 columns: Union[None, pd.Index, list, np.ndarray],
                 data=None):

        if data is None:
            data = self.data
        if columns is None:
            columns = data.columns

        if not decision_path:  # empty lists in python are considered False
            df = data.loc[:, columns]
        else:
            mask = " & ".join(decision_path).replace("<df>", "data")
            df = data.loc[eval(mask), columns]

        return df

    def find_split_points(self,
                          covariate_name: str,
                          decision_path: list):

        data = self.get_data(decision_path=decision_path, columns=[covariate_name])

        # TODO: only valid for numeric predictors. Write to handle categorical predictors too
        unique_points = np.sort(data.loc[:, covariate_name].unique())
        # find mid points and use them as split points
        split_points = (unique_points[1:] + unique_points[:-1]) / 2

        return split_points

    def impurity_reduction_constructors(self,
                                        split_points: np.ndarray,
                                        covariate_name: str,
                                        label: str,
                                        decision_path: list) -> tuple:

        # TODO: only valid for numeric predictors. Write to handle categorical predictors too
        # TODO: only valid for binary classification

        data = self.get_data(decision_path=decision_path, columns=[label, covariate_name])

        full_region_probability = len(data.loc[data[label] == 1, :].index) / len(data.index)

        left_observation_counts = []
        right_observation_counts = []
        left_region_probabilities = []
        right_region_probabilities = []
        for split_point in split_points:
            left_obs_count = len(data.loc[data[covariate_name] < split_point, :].index)
            right_obs_count = len(data.loc[data[covariate_name] >= split_point, :].index)

            left_p = len(data.loc[(data[label] == 1) & (data[covariate_name] < split_point), :].index)\
                / left_obs_count
            right_p = len(data.loc[(data[label] == 1) & (data[covariate_name] >= split_point), :].index)\
                / right_obs_count

            left_observation_counts.append(left_obs_count)
            right_observation_counts.append(right_obs_count)
            left_region_probabilities.append(left_p)
            right_region_probabilities.append(right_p)

        return full_region_probability, \
            np.array(left_observation_counts), np.array(right_observation_counts), \
            np.array(left_region_probabilities), np.array(right_region_probabilities)

    def check_if_pure(self,
                      label: str,
                      decision_path: list) -> bool:

        data = self.get_data(decision_path=decision_path, columns=None)

        # TODO: only valid for binary classification
        return (np.all(data.loc[:, label] == 1)) or (np.all(data.loc[:, label] == 0))

    def check_if_small(self,
                       min_obs_split: int,
                       decision_path: list):

        data = self.get_data(decision_path=decision_path, columns=None)

        return len(data.index) <= min_obs_split

    def node_misclassification_cost(self,
                                    node_key: str,
                                    decision_path: list,
                                    label: str) -> float:

        logging.debug(f"Computing misclassification cost for {node_key}")

        node_data = self.get_data(decision_path=decision_path, columns=None)
        fraction_obs_node = len(node_data.index) / len(self.data.index)
        # TODO: only valid for binary classification
        # if majority vote would classify observation as 1, then ...
        if len(node_data.loc[node_data[label] == 1, :].index) >= (len(node_data.index) // 2):
            fraction_misclassified_points = len(node_data.loc[node_data[label] == 0, :].index) \
                                            / len(node_data.index)
        else:
            fraction_misclassified_points = len(node_data.loc[node_data[label] == 1, :].index) \
                                            / len(node_data.index)

        logging.debug(
            f"fraction_obs_node: {fraction_obs_node}, fraction_misclassified_points: {fraction_misclassified_points}")

        return fraction_misclassified_points*fraction_obs_node

    @classmethod
    def predict_majority_class(cls,
                               data_leaves: list,
                               label: str,
                               return_new_data: bool) -> Union[np.ndarray, pd.DataFrame]:

        # predict with majority class under leaf and attach to corresponding new data
        predictions = []
        # TODO: only valid for binary classification
        for new_data_leaf, training_data_leaf in data_leaves:
            if not new_data_leaf.empty:  # only if there's some new_data under the leaf
                predicted_as_1 = training_data_leaf.loc[training_data_leaf[label] == 1, :]
                if len(predicted_as_1.index) >= (len(training_data_leaf.index) // 2):
                    new_data_leaf_predictions = [1] * len(new_data_leaf.index)
                else:
                    new_data_leaf_predictions = [0] * len(new_data_leaf.index)
                new_data_leaf.loc[:, label] = new_data_leaf_predictions
                predictions.append(new_data_leaf)

        # concatenate predictions and make sure to output them respecting order of new_data
        new_data_predictions = pd.concat(predictions, axis=0).sort_index()

        if return_new_data:
            return new_data_predictions
        else:
            return new_data_predictions[label].to_numpy()


class SQLHandler():

    def __init__(self,
                 impurity_function: Union[str, Callable]):

        self.impurity_function = impurity_function
        self.data = None

        self.server_connection = None
        self.connect()

    def connect(self):

        hostname = input("hostname: ")
        port = input("port (press Enter for default): ")
        username = getpass.getpass("username: ")  # does not show input when typing
        password = getpass.getpass("password: ")
        dbname = input("database: ")

        if port == "":
            port = "5432"  # default port

        print("Don't forget to close the connection when you are done")
        self.server_connection = psycopg2.connect(host=hostname, port=port,
                                                  user=username, password=password, dbname=dbname)

    def close_connection(self):

        if self.server_connection:
            self.server_connection.close()
            self.server_connection = None
            print("PostgreSQL server connection is now closed")

    def get_covariate_names(self,
                            label: str,
                            data=None):

        if data is None:
            # recall that self.data is the table name for SQLHandler
            data = self.data

        cursor = self.server_connection.cursor()
        query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = %(table_name)s; 
        """
        cursor.execute(query, {"table_name": data})
        columns = [column_name[0] for column_name in cursor.fetchall()]
        cursor.close()

        if label not in columns:
            return columns
        else:
            columns.remove(label)
            return columns

    @staticmethod
    def query_constructors(decision_path: list,
                           columns: list,
                           data: str):
        # in this case decision path has form [<column> <relation> <split_point>]
        # format query with placeholders and then use safe interpolation to inject parameters
        # placeholder for columns to select is simply column name
        # placeholder for where clauses is <column_name>_<relation>_<split_point>

        if not decision_path:
            columns_placeholders = []
            query_parameters = {"table_name": data}
            for column_name in columns:
                columns_placeholders.append(f"%({column_name})s")
                query_parameters[column_name] = column_name
            columns_placeholders = ", ".join(columns_placeholders)
            return columns_placeholders, None, query_parameters
        else:
            columns_placeholders = []
            where_placeholders = []
            query_parameters = {"table_name": data}
            for decision in decision_path:
                column_name, relation, split_point = decision.split(" ")
                if column_name in columns:  # only if requested to return
                    columns_placeholders.append(f"%({column_name})s")
                where_placeholders.append(f"%({column_name}_{relation}_{split_point})s")
                query_parameters.update({column_name: column_name,
                                         f"{column_name}_{relation}_{split_point}": decision})

            where_placeholders = " AND ".join(where_placeholders)
            columns_placeholders = ", ".join(columns_placeholders)
            return columns_placeholders, where_placeholders, query_parameters

    def get_data(self,
                 decision_path: list,
                 columns: Union[None, list],
                 data=None) -> pd.DataFrame:
        # we want to retrieve all data under a specific node, hence it makes sense to return Pandas DataFrames

        if data is None:
            data = self.data
        if columns is None:
            columns = data.columns

        cursor = self.server_connection.cursor()

        columns_placeholders, where_placeholders, query_parameters = \
            self.query_constructors(decision_path=decision_path, columns=columns, data=data)

        if not decision_path:  # empty lists in python are considered False
            query = f"""
                    SELECT {columns_placeholders}
                    FROM %(table_name)s; 
                    """
            cursor.execute(query, query_parameters)
        else:
            query = f"""
                    SELECT {columns_placeholders}
                    FROM %(table_name)s
                    WHERE {where_placeholders}; 
                    """
            cursor.execute(query, query_parameters)

        df = pd.DataFrame(cursor.fetchall(), columns=columns)
        cursor.close()
        return df

    def find_split_points(self,
                          covariate_name: str,
                          decision_path: list):
        pass

    def impurity_reduction_constructors(self,
                                        split_points: np.ndarray,
                                        covariate_name: str,
                                        label: str,
                                        decision_path: list) -> tuple:
        pass

    def check_if_pure(self,
                      label: str,
                      decision_path: list) -> bool:
        pass

    def check_if_small(self,
                       min_obs_split: int,
                       decision_path: list):
        pass

    def node_misclassification_cost(self,
                                    node_key: str,
                                    decision_path: list,
                                    label: str) -> float:
        pass

    @classmethod
    def predict_majority_class(cls,
                               data_leaves: list,
                               label: str,
                               return_new_data: bool) -> Union[np.ndarray, pd.DataFrame]:

        # predict method for trees has to output predictions and relies on get_data() (which queries the
        # database and returns a Pandas DataFrame, in turn), hence it makes sense to use the same data type here.
        # Moreover, this is reasonable also because get_data() is used by predict() in tree class only
        # to retrieve label values for training_data and also all observations for test_data, which implies
        # that is not too memory-intensive even for very large databases.

        # The same reasoning does not hold for find_split_points(), impurity_reduction_constructors(),
        # check_if_pure(), check_if_small() and node_misclassification_cost(), which would instead be too
        # memory-intensive using DataFrames and hence will perform operations directly on the database.

        return DataframeHandler.predict_majority_class(data_leaves=data_leaves,
                                                       label=label,
                                                       return_new_data=return_new_data)


