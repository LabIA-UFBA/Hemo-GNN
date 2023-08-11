'''
Class to read hemophilia data - features and graph.
'''
import warnings
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import StratifiedKFold

from loguru import logger as log

warnings.simplefilter("ignore")


class Hemophilia:

    def __init__(self, gf, ff, fn,
                       ab = 'AA_HGVS', 
                       at = 'Calculated_Severity',
                       k=10,
                       inverse = False, seed=None) -> None:
        '''
        Dataset Hemophilia - features and graph

        Attributes:
          gf (str) - graph filename
        '''
        ini_time = time.time()
        log.info('read, processind and cread folds...')
        self.meta = self.__process_data(filename_graph=gf, 
                                        filename_data=ff, 
                                        feature_names=fn, 
                                        att_bind=ab, 
                                        att_target=at, 
                                        FOLDS=k,
                                        inverse=inverse, 
                                        seed=None)
        
        self.node_features, self.edges, self.edge_weights = self.meta['graph_info']
        self.edges_COO = self.meta['edges_COO']
        self.x = self.meta['x']
        self.y = self.meta['y']
        self.folds = self.meta['folds']
        self.num_classes = self.meta['num_class']

        duration = str(timedelta(seconds=(time.time()-ini_time))).split('.')[0]
        log.info(f'Done, processing in {duration}.')


    def __convert_indexes_w(self, edges, dic):
        
        new_edge = np.zeros((edges.shape[0], 2), dtype=int)
        for i in np.arange(new_edge.shape[0]):
            s = dic.get(edges.iloc[i,0])
            t = dic.get(edges.iloc[i,1])
            new_edge[i,:] = np.array((s, t))

        data = {'source': new_edge[:,0], 'target': new_edge[:,1], 'w': edges['weight']}
            
        return pd.DataFrame(data)


    def __process_data(self, filename_graph, 
                                        filename_data, 
                                        feature_names, 
                                        att_bind, 
                                        att_target, 
                                        FOLDS,
                                        inverse, seed=None):

        # graph
        graph = pd.read_csv(filename_graph, sep="\t")
        graph = graph.rename({'pos1':'source', 'pos2': 'target'}, axis='columns')
        #graph.drop(['weight'], inplace=True, axis=1)

        # features
        features = pd.read_csv(filename_data, sep="\t")
        #features.drop(['Protein_Change'], inplace=True, axis=1)
        #features = features.fillna(0.1)
        # drop and reset index - fix bug
        features.dropna(inplace=True)
        features.reset_index(drop=True, inplace=True)

        # sincronizar os indices
        node_index_dict = {}
        node_index_dic = {}
        indice = 0
        for node in features[att_bind].unique():
            indexs = list(features[features[att_bind]==node].index)
            node_index_dict[node] = indexs[0]
            node_index_dic[node] = indice
            indice += 1

        del indice

        # train data
        data_train = features.iloc[list(node_index_dict.values())]

        # get size minority class
        size_minority = min(Counter(data_train[att_target]).values())
        
        log.info(f'Class structure: {Counter(data_train[att_target])}')

        # get edges
        edges = graph.iloc[np.where(graph.source.isin(data_train[att_bind]))]
        edges = edges.iloc[np.where(edges.target.isin(data_train[att_bind]))]
        
        #data_train['w'] = edge['weight']

        # convert node values to range index data train
        edge = self.__convert_indexes_w(edges, node_index_dic)

        for i in np.arange(data_train.shape[0]):
            data_train.iloc[i, 0] = node_index_dic.get(data_train.iloc[i, 0])

        classes = list(set(data_train[att_target]))
        #feature_names = set(data_train.columns) - {att_target, att_bind, 'AA_dist', 'areaSES'}

        classes_numerical = np.arange(len(classes))

        data_train[att_target].replace(classes, classes_numerical, inplace=True)
        
        # stand
        #for feature in feature_names:
        #    data_train[feature] = stand(data_train[feature])

        # norm 0-1
        #for feature in feature_names:
        #    data_train[feature] = norm_min_max(data_train[feature])

        # undersampling to create balanced dataset
        subset = []
        for classe in classes_numerical:
            subset.append(data_train[data_train[att_target] == classe][:size_minority])
        
        #data_train_balanced = data_train
        
        # new split

        #
        # replace
        #
        skf = StratifiedKFold(n_splits=FOLDS, random_state=seed, shuffle=False)

        folds = []

        for train_index, test_index in skf.split(data_train[feature_names], data_train[att_target]):
            # train - 
            x_train = data_train[att_bind].iloc[train_index].to_numpy() # train_index (X MAX) == (10 MIN)
            y_train = data_train[att_target].iloc[train_index].to_numpy()
            
            x_test = data_train[att_bind].iloc[test_index].to_numpy()
            y_test = data_train[att_target].iloc[test_index].to_numpy()
            
            folds.append([x_train, y_train, x_test, y_test])
        
        x_train, y_train, x_test, y_test = folds[0]
        log.info(f'train: {y_train.shape}, test: {y_test.shape}')
        log.info('-' * 30)

        # Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
        edges = edge[["source", "target"]].to_numpy()

        # Create an edge weights array of ones.
        #edge_weights = tf.ones(shape=edges.shape[1])
        
        if inverse:
            edge_weights = 1/edge['w'].to_numpy()
        else:
            edge_weights = edge['w'].to_numpy()

        # Create a node features array of shape [num_nodes, num_features].
        node_features = tf.cast(
            data_train.sort_values(att_bind)[feature_names].to_numpy(), dtype=tf.dtypes.float32
        )
        
        node_states = tf.convert_to_tensor(data_train.sort_values(att_bind)[feature_names])
        
        return {'graph_info': [node_features, edges, edge_weights], 
                'edges_COO': node_states,
                'x': data_train.sort_values(att_bind)[feature_names].to_numpy(), 
                'y': data_train.sort_values(att_bind)[att_target].to_numpy(),
                'folds': folds, 
                'num_class': len(classes)}