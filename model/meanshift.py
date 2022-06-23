# mean shift model
# try to cluster data in periodic real space
# or latent space with weights
import numpy as np


def fit(data, h, periodic=None, weights=None):
    if periodic is None:
        d = np.max(np.max(data, axis=1) - np.min(data, axis=1)) * 1e-3
    else:
        d = periodic * 1e-3

    Group_Center_Distance = d
    centers, current_labels = data, np.arange(data.shape[0])
    for iter in range(3):
        Stop_Shift_Distance = d * 10 ** (-2 - iter)
        centers = shift_vectors(
            centers, data, h, Stop_Shift_Distance, periodic=periodic, weights=weights)
        # <labels> size = <input_centers> size
        # <labels> like [1, 0, 1, n-1, ...], n is <output_centers> size
        values = points_potential(centers, data, h, periodic, weights)
        centers, labels = merge_centers_with_values(
            centers, values, Group_Center_Distance, periodic=periodic, use_mean=True)

        _labels = np.zeros_like(current_labels)
        for i, index in enumerate(labels):
            _labels[current_labels == i] = index
        current_labels = _labels

    # sort centers by its item number
    _, counts = np.unique(current_labels, return_counts=True)
    args = np.argsort(counts)[::-1]
    centers = centers[args, :]
    _labels = np.zeros_like(current_labels)
    for i, index in enumerate(args):
        _labels[current_labels == index] = i
    current_labels = _labels

    return centers, current_labels


def gaussian_kernel(vectors, bandwidth):
    # reduce the last index
    dx = np.linalg.norm(vectors, axis=-1) / bandwidth
    return np.exp(dx ** 2 / -2)


def single_point_step(point, points, kernel_bandwidth, periodic=None, weights=None):
    vectors = points - point
    if periodic is not None:
        vectors = (vectors + periodic / 2) % periodic - periodic / 2

    point_weights = gaussian_kernel(vectors, kernel_bandwidth)

    if weights is not None:
        point_weights = point_weights * weights

    dx = np.matmul(point_weights, vectors) / np.sum(point_weights)
    return dx


def points_potential(points, data, h, periodic=None, weights=None):
    vectors = points.reshape(points.shape[0], 1, points.shape[1]) - data
    # same as:
    #   vectors = np.array([data - point for point in points])
    if periodic is not None:
        vectors = (vectors + periodic / 2) % periodic - periodic / 2

    point_weights = gaussian_kernel(vectors, h)

    if weights is not None:
        return np.matmul(point_weights, weights)
    else:
        return np.sum(point_weights, axis=1)


def shift_vectors(vectors, data, h, error, periodic=None, weights=None):
    new_v = []
    for v in vectors:
        _v = v
        while True:
            dv = single_point_step(
                _v, data, h, periodic=periodic, weights=weights)
            if np.linalg.norm(dv) < error:
                break
            _v = _v + dv

        new_v.append(_v)
    if periodic is None:
        return np.array(new_v)
    else:
        return np.array(new_v) % periodic


def merge_centers(centers, error, periodic=None):
    new_centers = np.array(centers[0]).reshape(1, -1)
    new_labels = None
    while True:
        len_new_centers = new_centers.shape[0]
        new_labels = np.zeros((centers.shape[0], ), dtype=np.int)
        # merge centers
        for i, center in enumerate(centers):
            dx = new_centers - center
            if periodic is not None:
                dx = (dx + periodic / 2) % periodic - periodic / 2
            norm = np.linalg.norm(dx, axis=1)
            a = np.argmin(norm)
            if norm[a] < error:
                n = np.argwhere(new_labels == i).size
                new_centers[a] = (new_centers[a] * n + center) / (n + 1)
                new_labels[i] = a
            else:
                new_labels[i] = new_centers.shape[0]
                new_centers = np.vstack((new_centers, center.reshape(1, -1)))

        if periodic is not None:
            new_centers = new_centers % periodic

        if len_new_centers == new_centers.shape[0]:
            break

    return new_centers, new_labels


def merge_centers_with_values(centers, values, error, periodic=None, use_mean=False):
    # merge centers with their values, sort by values and filter range < error
    args = np.argsort(values).reshape(-1)
    new_centers = []
    new_labels = [-1] * args.size
    unlabel = list(args)
    index = 0
    while len(unlabel) > 0:
        i = unlabel[0]
        new_centers.append(centers[i, :].reshape(-1))

        dx = centers - centers[i, :]
        if periodic is not None:
            dx = (dx + periodic / 2) % periodic - periodic / 2
        norm = np.linalg.norm(dx, axis=1)
        neighbors = np.argwhere(norm < error).reshape(-1)

        for j in neighbors:
            if j in unlabel:
                unlabel.remove(j)
                new_labels[j] = index

        index += 1
    new_centers = np.array(new_centers)
    new_labels = np.array(new_labels)

    if use_mean:
        for i in range(max(new_labels) + 1):
            new_centers[i] = np.mean(centers[new_labels == i], axis=0)

    return new_centers, new_labels


def select_data(data, error, periodic):
    # center must sorted by energy
    A = list(range(data.shape[0]))
    B = []
    while len(A) > 0:
        a = A[0]
        dx = max_diff(data, data[a, :], periodic)
        C = list(np.argwhere(dx < error).reshape(-1))

        B.append(a)
        for x in C:
            if x in A:
                A.remove(x)

    return B
