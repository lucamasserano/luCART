import pytest
import pandas as pd
import numpy as np

from luCART.classification_tree import tree


def random_data(seed=7, size=200):

    np.random.seed(seed)
    two_dim_normal = pd.DataFrame(
        np.random.multivariate_normal(mean=[4, 8], cov=np.array([[2, 0], [0, 2]]), size=size//2))
    two_dim_normal2 = pd.DataFrame(
        np.random.multivariate_normal(mean=[8, 4], cov=np.array([[2, 0], [0, 2]]), size=size//2))
    two_dim_normal["y"] = 0
    two_dim_normal2["y"] = 1
    synthetic_data = pd.concat([two_dim_normal, two_dim_normal2], axis=0, ignore_index=True)
    synthetic_data.rename(columns={0: "0", 1: "1"}, inplace=True)

    return synthetic_data


def generate_example_data():
    example_data = pd.DataFrame({"age": [45, 67, 12, 34, 2, 9, 56], "temperature": [23, 34, 12, 28, 2, 9, 40],
                                "weight": [45, 67, 100, 84, 2, 67, 56]})
    return example_data


def generate_example_tree():
    test_tree = tree.ClassificationTree(impurity_function="gini", data_source_type="dataframe")
    test_tree.add_node(parent_node=None,
                       new_node=tree.Node(parent_split_covariate="root", parent_split_point="root",
                                          parent_split_relation="root"),
                       where=None)
    test_tree.add_node(parent_node=test_tree.root,
                       new_node=tree.Node(parent_split_covariate="age", parent_split_point=30,
                                          parent_split_relation="<"),
                       where="left")
    test_tree.add_node(parent_node=test_tree.root.left,
                       new_node=tree.Node(parent_split_covariate="temperature", parent_split_point=20,
                                          parent_split_relation="<"),
                       where="left")
    test_tree.add_node(parent_node=test_tree.root,
                       new_node=tree.Node(parent_split_covariate="age", parent_split_point=30,
                                          parent_split_relation=">="),
                       where="right")
    test_tree.add_node(parent_node=test_tree.root.left,
                       new_node=tree.Node(parent_split_covariate="temperature", parent_split_point=20,
                                          parent_split_relation=">="),
                       where="right")
    test_tree.add_node(parent_node=test_tree.root.right,
                       new_node=tree.Node(parent_split_covariate="weight", parent_split_point=65,
                                          parent_split_relation="<"),
                       where="left")

    return test_tree


def test_check_relation():

    with pytest.raises(ValueError):
        tree.Node(parent_split_covariate="covariate",
                  parent_split_relation="wrong",
                  parent_split_point=10)


def test_add_node():

    example_tree = generate_example_tree()
    assert example_tree.tree_cardinality == 6
    assert len(example_tree.leaves) == 3
    assert len(example_tree.keys) == 6
    assert example_tree.root.key == "root"
    assert example_tree.root.children_keys == {'age < 30': 'left',
                                               'temperature < 20': 'left',
                                               'age >= 30': 'right',
                                               'temperature >= 20': 'left',
                                               'weight < 65': 'right'}


def test_prune_at_node():

    example_tree = generate_example_tree()
    example_tree.prune_at_node(subtree_root_key="age < 30")
    assert example_tree.tree_cardinality == 4
    assert example_tree.keys == {'age < 30', 'age >= 30', 'root', 'weight < 65'}
    assert len(example_tree.leaves) == 2
    assert example_tree.root.children_keys == {'age < 30': 'left', 'age >= 30': 'right', 'weight < 65': 'right'}


def test_search_node():
    pass


def test_decision_path():

    example_tree = generate_example_tree()

    assert example_tree.decision_path(target_node=example_tree.root.left.left) == \
        ["(<df>['age'] < 30)", "(<df>['temperature'] < 20)"]


def test_get_data():

    example_tree = generate_example_tree()
    example_data = generate_example_data()

    retrieved_data = example_tree.get_data(target_node=example_tree.root.left.left, data=example_data)
    check = pd.DataFrame({"age": [12, 2, 9], "temperature": [12, 2, 9], "weight": [100, 2, 67]})
    assert retrieved_data.reset_index(drop=True).equals(check)


def test_covariate_impurity_reduction():
    pass


def test_choose_split():
    pass


def test_build_tree():
    pass


def test_node_misclassification_cost():
    pass


def test_prune_node_cost_increase():
    pass


def test_pruning_step():
    pass


def test_prune_tree():
    pass


def test_fit():

    synthetic_data = random_data()

    # without pruning
    test_tree = tree.ClassificationTree(impurity_function="gini", data_source_type="dataframe")
    test_tree.fit(data=synthetic_data, label="y")

    assert test_tree.tree_cardinality == 9
    assert len(test_tree.leaves) == 5
    assert test_tree.keys == {'0 < 4.452409599160356', '0 < 7.548204542527349', '0 >= 4.452409599160356',
                              '0 >= 7.548204542527349', '1 < 3.308528430779547', '1 < 5.7421119018332965',
                              '1 >= 3.308528430779547', '1 >= 5.7421119018332965', 'root'}
    assert test_tree.root.children_keys == {'1 < 5.7421119018332965': 'left', '1 >= 5.7421119018332965': 'right',
                                            '0 < 4.452409599160356': 'left', '0 >= 4.452409599160356': 'left',
                                            '1 < 3.308528430779547': 'left', '1 >= 3.308528430779547': 'left',
                                            '0 < 7.548204542527349': 'right', '0 >= 7.548204542527349': 'right'}

    # with pruning
    test_tree = tree.ClassificationTree(impurity_function="gini", alpha=0.01, data_source_type="dataframe")
    test_tree.fit(data=synthetic_data, label="y")

    assert test_tree.tree_cardinality == 7
    assert len(test_tree.leaves) == 4
    assert test_tree.keys == {'0 < 4.452409599160356', '0 < 7.548204542527349', '0 >= 4.452409599160356',
                              '0 >= 7.548204542527349', '1 < 5.7421119018332965', '1 >= 5.7421119018332965', 'root'}
    assert test_tree.root.children_keys == {'1 < 5.7421119018332965': 'left', '1 >= 5.7421119018332965': 'right',
                                            '0 < 4.452409599160356': 'left', '0 >= 4.452409599160356': 'left',
                                            '0 < 7.548204542527349': 'right', '0 >= 7.548204542527349': 'right'}


def test_predict():

    synthetic_data = random_data()
    test_data = random_data(seed=1)
    test_tree = tree.ClassificationTree(impurity_function="gini", alpha=1, data_source_type="dataframe")
    test_tree.fit(data=synthetic_data, label="y")

    predictions = test_tree.predict(new_data=test_data.drop(labels="y", axis=1))

    assert np.sum(predictions != test_data["y"])/len(predictions) - 0.095 < 1e-10
    assert np.all(predictions == np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
                                           1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                                           1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                                           1, 0]))


def test_choose_alpha():

    synthetic_data = random_data()
    test_tree = tree.ClassificationTree(impurity_function="gini", data_source_type="dataframe",
                                        alpha=[0.00001, 0.0001, 0.001, 0.01],
                                        n_folds=5, debug=True)
    test_tree.fit(data=synthetic_data, label="y")

    assert test_tree.alpha_cv == 1e-05



