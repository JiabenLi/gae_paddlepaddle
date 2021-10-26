# from __future__ import division
# from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import paddle

from model import GCNModelVAE
from utils import load_data, mask_test_edges_eval, preprocess_graph, get_roc_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--data_path', type=str, default='', help='path of data.')

args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str, args.data_path)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_eval(adj, args.data_path)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)

    model.load_dict(paddle.load(args.data_path + '/gae_paddlepaddle.pdparams'))
    recovered, mu, logvar = model(features, adj_norm)
    hidden_emb = mu.numpy()

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)