import os
import pickle

import numpy as np

# Explicitely declare the mapping to use as reference.
ix2label = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'}
label2ix = {v: k for k, v in ix2label.items()}

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def reshape(ima_row):
    ima = ima_row.reshape([32, 32, 3], order='F')
    ima = np.transpose(ima, (1, 0, 2))
    return ima

def process_batch(data_batch, target_ixlabels):
    new_data_batch = {
        'images': [],
        'labels': []}

    label_map = {ix: i for i, ix in enumerate(target_ixlabels)}

    for i, true_label in enumerate(data_batch[b'labels']):
        if true_label not in target_ixlabels:
            continue

        new_data_batch['images'].append(reshape(data_batch[b"data"][i]))
        new_data_batch['labels'].append(label_map[true_label])

    return new_data_batch


def match_priors(images, labels, prior):
    new_images, new_labels = [], []
    # Normalize to more frequent label
    prior = np.array(prior)/np.max(prior)

    for label in range(len(prior)):
        idx = np.where(labels == label)[0]
        n = len(idx)
        idx = idx[np.random.permutation(n)[:int(n*prior[label])]]

        new_images += images[idx].tolist()
        new_labels += labels[idx].tolist()

    # Convert to numpy array and shuffle
    perm = np.random.permutation(len(new_images))

    images = np.array(new_images)[perm]
    labels = np.array(new_labels)[perm]

    return images, labels

def get_cifar10(data_path, test=False, prior=None, test_prior=None):
    cifar10 = {
        'images': [],
        'labels': []}

    train_batches = ["data_batch_{}".format(i+1) for i in range(5)]
    target_ixlabels = list(range(10))

    for batch in train_batches:
        data_batch = unpickle(os.path.join(data_path, batch))
        curr_batch = process_batch(data_batch, target_ixlabels)

        cifar10['images'] += curr_batch['images']
        cifar10['labels'] += curr_batch['labels']

    # Convert to numpy array
    cifar10['images'] = np.array(cifar10['images'])
    cifar10['labels'] = np.array(cifar10['labels'])

    # Adjust dataset to meet the specified priors
    if prior is not None:
        imas, labels = match_priors(
                cifar10['images'],
                cifar10['labels'],
                prior)
        cifar10['images'], cifar10['labels'] = imas, labels

    if test:
        test_batch = unpickle(os.path.join(data_path, 'test_batch'))
        test_batch = process_batch(test_batch, target_ixlabels)

        cifar10['test_images'] = test_batch['images']
        cifar10['test_labels'] = test_batch['labels']

        # Convert to numpy array
        cifar10['test_images'] = np.array(cifar10['test_images'])
        cifar10['test_labels'] = np.array(cifar10['test_labels'])

        # Adjust test set to meet the specified priors
        if test_prior is not None:
            imas, labels = match_priors(
                    cifar10['test_images'],
                    cifar10['test_labels'],
                    test_prior)
            cifar10['test_images'], cifar10['test_labels'] = imas, labels

    return cifar10, ix2label
