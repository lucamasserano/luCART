import numpy as np

from luCART.tests.test_tree import random_data
from luCART.random_forest import random_forest


def test_choose_split():
    pass


def test_fit():

    synthetic_data = random_data()
    test_random_forest = random_forest.RandomForestClassifier(n_classifiers=10,
                                                              n_covariates="sqrt",
                                                              impurity_function="gini",
                                                              data_source_type="dataframe",
                                                              debug=True)
    test_random_forest.fit(data=synthetic_data, label="y")

    assert len(test_random_forest.forest) == 10
    cardinalities = []
    for key, tree in test_random_forest.forest.items():
        cardinalities.append(tree["tree_cardinality"])
    assert cardinalities == [33]*10

    assert test_random_forest.tree_cardinality == 0
    assert test_random_forest.root is None
    assert test_random_forest.leaves == {}
    assert test_random_forest.keys == set()


def test_rf_predict():

    synthetic_data = random_data()
    test_data = random_data(seed=1)

    test_random_forest = random_forest.RandomForestClassifier(n_classifiers=10,
                                                              n_covariates="sqrt",
                                                              impurity_function="gini",
                                                              data_source_type="dataframe",
                                                              debug=True)
    test_random_forest.fit(data=synthetic_data, label="y")
    predictions = test_random_forest.rf_predict(new_data=test_data.drop(labels="y", axis=1))
    assert np.sum(predictions != test_data["y"])/len(predictions) - 0.12 < 1e-10
    assert np.all(predictions == np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                                           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                                           1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
                                           1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           0, 1]))
