import torch
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce



#Parallel
from functools import partial
import concurrent.futures

import os
import deepdish as dd
import networkx as nx
import numpy as np

# Extra functions
from Imports.gdc import GDC

def list_files_of_type(directory, file_type):
    files = os.listdir(directory)

    filtered_files = [file for file in files if file.endswith(file_type)]

    return filtered_files


def read_single_data(filename, data_dir, use_gdc=False):
    temp = dd.io.load(os.path.join(data_dir, filename))

    # read edge and edge attribute
    S_conn = temp['S_conn'][()]
    num_nodes = S_conn.shape[0]

    # Graph with edges
    G = nx.from_numpy_array(S_conn)
    A = nx.to_scipy_sparse_array(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = S_conn[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    # Feature
    att = temp['F_corr'][()]
    att = np.arctanh(att)  # normalization
    # Target class: sex
    info = temp['label']
    if info['Gender'].values == 'M':
        label = np.array([1])
    else:
        label = np.array([0])

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(), data.edge_index.data.numpy(), data.x.data.numpy(), data.y.data.item(), num_nodes

    else:
        return edge_att.data.numpy(), edge_index.data.numpy(), att, label, num_nodes


def read_data(data_dir):
    onlyfiles = [f for f in list_files_of_type(data_dir, '.h5')]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list, att_list = [], [], []

    # parallel computing: method I
    import timeit
    start = timeit.default_timer()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        partial_list_locations = partial(read_single_data, data_dir=data_dir)
        res = list(executor.map(partial_list_locations, onlyfiles))

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # # parallel computing: method II
    # cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=cores)
    # #pool =  MyPool(processes = cores)
    # func = partial(read_single_data, data_dir=data_dir)
    # import timeit
    # start = timeit.default_timer()
    # res = pool.map(func, onlyfiles)
    # pool.close()
    # pool.join()
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1] + j * res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j] * res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos=pseudo_torch)

    data, slices = split(data, batch_torch)

    return data, slices


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


class Own_database_create(InMemoryDataset):

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(Own_database_create, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # data_dir = osp.join(self.root,'raw')
        data_dir = self.root
        onlyfiles = [f for f in list_files_of_type(data_dir, '.h5')]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        return  self.name+'.pt'

    def download(self):
        # Download to `self.raw_dir`.
        # return
        pass

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices = read_data(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


