
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import h5py
import pandas as pd
import pdb

def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range
    Parameters
    ----------
    data : np.int32 arrays
    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data
    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)

    return data, id_dict, n

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out

def load_data_from_database(dataset, mode='transductive', testing=False, rating_map=None, post_rating_map=None, ratio=1.0):
    """
    Loads official train/test split and uses 10% of training samples for validaiton
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
    """

    dtypes = {
        'u_nodes': np.str, 'v_nodes': np.int32,
        'ratings': np.float32}

    filename_train = 'data/' + dataset + '/' + mode + '/train.csv'
    filename_test = 'data/' + dataset + '/' + mode + '/test.csv'

    data_train = pd.read_csv(
        filename_train, header=None,
        names=['u_nodes', 'v_nodes', 'ratings'], dtype=dtypes)

    data_test = pd.read_csv(
        filename_test, header=None,
        names=['u_nodes', 'v_nodes', 'ratings'], dtype=dtypes)

    if mode == 'inductive':
        filename_test_init = 'data/' + dataset + '/' + mode + '/test_init.csv'

        data_test_init = pd.read_csv(
            filename_test_init, header=None,
            names=['u_nodes', 'v_nodes', 'ratings'], dtype=dtypes)

    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)

    if mode == 'inductive':
        data_array_test_init = data_test_init.values.tolist()
        data_array_test_init = np.array(data_array_test_init)

    if ratio < 1.0:
        data_array_train = data_array_train[data_array_train[:, -1].argsort()[:int(ratio*len(data_array_train))]]

    if mode == 'inductive':
        data_array = np.concatenate([data_array_train, data_array_test, data_array_test_init], axis=0)
    else:
        data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])

    ratings = data_array[:, 2].astype(dtypes['ratings'])
    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1
    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])
    
    # number of test and validation edges, see cf-nade code
    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    if mode == 'inductive':
        num_test_init = data_array_test_init.shape[0]

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])


    for i in range(len(ratings)):
        assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])


    if mode == 'inductive':
        idx_nonzero_train = idx_nonzero[0:num_train+num_val]
        idx_nonzero_test = idx_nonzero[num_train+num_val:num_train+num_val+num_test]
        idx_nonzero_test_init = idx_nonzero[num_train+num_val+num_test:]

        pairs_nonzero_train = pairs_nonzero[0:num_train+num_val]
        pairs_nonzero_test = pairs_nonzero[num_train+num_val:num_train+num_val+num_test]
        pairs_nonzero_test_init = pairs_nonzero[num_train+num_val+num_test:]

    else:

        idx_nonzero_train = idx_nonzero[0:num_train+num_val]
        idx_nonzero_test = idx_nonzero[num_train+num_val:]

        pairs_nonzero_train = pairs_nonzero[0:num_train+num_val]
        pairs_nonzero_test = pairs_nonzero[num_train+num_val:]


    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    if mode == 'inductive':
        idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test, idx_nonzero_test_init], axis=0)
        pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test, pairs_nonzero_test_init], axis=0)

        val_idx = idx_nonzero[0:num_val]
        train_idx = idx_nonzero[num_val:num_train + num_val]
        test_idx = idx_nonzero[num_train + num_val:num_train + num_val + num_test]
        test_init_idx = idx_nonzero[num_train + num_val + num_test:]

    else:
        idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
        pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

        val_idx = idx_nonzero[0:num_val]
        train_idx = idx_nonzero[num_val:num_train + num_val]
        test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:num_train + num_val + num_test]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        if mode == 'inductive':
            train_idx = np.hstack([train_idx, val_idx, test_init_idx])
        else:
            train_idx = np.hstack([train_idx, val_idx])
    
    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    # Features
    # drug features
    drugs_file = 'data/' + dataset + '/drugs_features.csv'
    genes_file = 'data/' + dataset + '/genes_features.csv'

    if os.path.exists(drugs_file) and os.path.exists(genes_file):
        drugs_df = pd.read_csv(drugs_file)

        drugs_headers = drugs_df.columns.values[:-1]
        num_drug_features = drugs_headers.shape[0]

        u_features = np.zeros((num_users, num_drug_features), dtype=np.float32)
        for drugbank_id, d_vec in zip(drugs_df['drugbank_id'].values.tolist(), drugs_df[drugs_headers].values.tolist()):
            # check if drugbank_id was listed in ratings file and therefore in mapping dictionary
            if drugbank_id in u_dict.keys():
                u_features[u_dict[drugbank_id], :] = d_vec

        # gene features
        genes_df = pd.read_csv(genes_file)

        genes_headers = genes_df.columns.values[:-1]
        num_gene_features = genes_headers.shape[0]

        v_features = np.zeros((num_items, num_gene_features), dtype=np.float32)
        for gene_id, g_vec in zip(genes_df['gene_id'].values.tolist(), genes_df[genes_headers].values.tolist()):
            # check if gene_id was listed in ratings file and therefore in mapping dictionary
            if gene_id in v_dict.keys():
                v_features[v_dict[gene_id], :] = g_vec

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)

        print("Drug features shape: "+str(u_features.shape))
        print("Gene features shape: "+str(v_features.shape))

    else:
        u_features = None
        v_features = None

    drug_dict = {v: k for k, v in u_dict.items()}
    gene_dict = {val: key for key,val in v_dict.items()}

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, drug_dict, gene_dict