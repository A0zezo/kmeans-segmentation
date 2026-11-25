import numpy as np
from kmeans import kmeans, make_customers


def test_separates_two_blobs():
    X = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 10])
    lab, _, inertia = kmeans(X, 2)
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1 and lab[0] != lab[10]
    assert inertia == 0


def test_label_range():
    lab, C, _ = kmeans(make_customers(), 3)
    assert set(lab) <= {0, 1, 2} and C.shape == (3, 3)


def test_deterministic():
    a = kmeans(make_customers(), 3)[2]
    assert a == kmeans(make_customers(), 3)[2]
