import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
from sklearn import preprocessing
from torch_geometric.data import Dataset, Data
import torch_geometric_temporal as pygt

class ImputationDataset(Dataset):
    def __init__(self, root='', base_percent=0.8, train_percent=0.87, val_percent=0.94, test_percent=0.97, seed=2,
                 predict_in=6, past_horizon=20, do_time_split=False, fully_observed=False):
        super(ImputationDataset, self).__init__(root)
        self.test_percent = test_percent
        self.base_percent = base_percent
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.predict_in = predict_in
        self.past_horizon = past_horizon
        self.coords = None
        self.node_ids = None
        self.fully_observed = fully_observed

        self.init_data()
        self.init_seeds(seed)
        self.set_masks()
        if do_time_split:
            self.test_count = round(self.x.shape[0] * 0.25)
            self.val_count = round(self.x.shape[0] * 0.25)
        else:
            self.val_count = 0
            self.test_count = 0
        # self.norm_with_masks(self.base_mask + self.train_mask)

    def init_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def len(self) -> int:
        return len(self.x) - self.past_horizon - self.predict_in - self.val_count - self.test_count

    def shuffle_masks(self):
        mask = self.base_mask + self.train_mask
        idxs = torch.where(mask)[0][torch.randperm(torch.sum(mask))]
        base_idxs = idxs[:torch.div(torch.sum(mask), 2, rounding_mode='trunc')]
        train_idxs = idxs[torch.div(torch.sum(mask), 2, rounding_mode='trunc'):]
        self.base_mask = torch.zeros_like(mask)
        self.base_mask[base_idxs] = True
        self.train_mask = torch.zeros_like(mask)
        self.train_mask[train_idxs] = True
        # self.x, self.node_labels = self.norm_with_masks(self.base_mask + self.train_mask)

    def set_masks(self):
        num_nodes = self.x.shape[1]
        idxs = torch.randperm(num_nodes)
        self.test_mask = self.get_mask(idxs, 0, round(self.test_percent * num_nodes))
        self.val_mask = self.get_mask(idxs, round(self.test_percent * num_nodes), round(self.val_percent * num_nodes))
        self.train_mask = self.get_mask(idxs, round(self.train_percent * num_nodes),
                                        round(self.base_percent * num_nodes))
        self.base_mask = self.get_mask(idxs, round(self.base_percent * num_nodes), num_nodes)
        self.mask_enc = (self.test_mask * 1 + self.val_mask * 2 + self.train_mask * 3 + self.base_mask * 4).numpy()

    def get_mask(self, idxs, start, stop):
        test_idxs = idxs[start:stop]
        zero_tensor = torch.zeros_like(idxs, dtype=torch.bool)
        zero_tensor[test_idxs] = True
        return zero_tensor

    def get(self, idx):
        # self.shuffle_masks()
        return Data(x=self.x[idx:idx + self.past_horizon],
                    y=self.x[idx + self.predict_in:idx + self.past_horizon + self.predict_in],
                    edge_index=self.edge_index,
                    base_mask=self.base_mask,
                    train_mask=self.train_mask, val_mask=self.val_mask, test_mask=self.test_mask,
                    attributes=self.node_labels, coords=self.coords,
                    node_ids=self.node_ids)  # mean=mean,                    standard_deviation=standard_deviation
        # return self.data[idx]

    def get_train(self):
        return Data(x=self.x[:self.train_end() - self.predict_in],
                    y=self.x[self.predict_in:self.train_end()],
                    edge_index=self.edge_index,
                    base_mask=self.base_mask,
                    train_mask=self.train_mask, val_mask=self.val_mask, test_mask=self.test_mask,
                    attributes=self.node_labels, coords=self.coords, node_ids=self.node_ids, mask_enc=self.mask_enc)

    def get_val(self):
        return Data(x=self.x[-self.val_count - self.test_count:self.x.shape[0] - self.predict_in - self.test_count],
                    y=self.x[-self.val_count - self.test_count + self.predict_in:self.x.shape[0]-self.test_count],
                    edge_index=self.edge_index,
                    base_mask=self.base_mask,
                    train_mask=self.train_mask, val_mask=self.val_mask, test_mask=self.test_mask,
                    attributes=self.node_labels, coords=self.coords, node_ids=self.node_ids, mask_enc=self.mask_enc)

    def get_test(self):
        return Data(x=self.x[-self.test_count:self.x.shape[0] - self.predict_in],
                    y=self.x[-self.test_count + self.predict_in:],
                    edge_index=self.edge_index,
                    base_mask=self.base_mask,
                    train_mask=self.train_mask, val_mask=self.val_mask, test_mask=self.test_mask,
                    attributes=self.node_labels, coords=self.coords, node_ids=self.node_ids, mask_enc=self.mask_enc)

    def train_end(self):
        return self.x.shape[0] - self.val_count - self.test_count


class MetrLALoader(ImputationDataset):

    def __init__(self, base_percent=0.8, train_percent=0.87, val_percent=0.94, test_percent=0.97, predict_in=6,
                 past_horizon=20, seed=2, do_time_split=False):
        super(MetrLALoader, self).__init__(base_percent=base_percent, train_percent=train_percent,
                                           val_percent=val_percent, test_percent=test_percent, predict_in=predict_in,
                                           past_horizon=past_horizon, seed=seed, do_time_split=do_time_split)

    def init_data(self):
        dataloader = pygt.dataset.METRLADatasetLoader()
        full_data = dataloader.get_dataset(num_timesteps_in=34272, num_timesteps_out=0)
        self.edge_index = torch.LongTensor(full_data.edge_index)
        self.edge_weight = torch.FloatTensor(full_data.edge_weight)
        self.x = torch.FloatTensor(full_data.features).squeeze().permute(2, 0, 1)[:, :, :1]  # [:1000]
        self.node_labels = None


class SumoTrafficDataset(ImputationDataset):
    def __init__(self, root='', base_percent=0.8, train_percent=0.87, val_percent=0.94, test_percent=0.97,
                 reload=False, seed=2, predict_in=6, past_horizon=20, do_time_split=False, fully_observed=False):

        self.test_percent = test_percent
        self.base_percent = base_percent
        self.train_percent = train_percent
        self.val_percent = val_percent

        if (reload):
            try:
                os.remove(self.processed_paths[0])
            except:
                pass

        super(SumoTrafficDataset, self).__init__(root, base_percent, train_percent, val_percent, test_percent,
                                                 predict_in=predict_in, seed=seed, past_horizon=past_horizon,
                                                 do_time_split=do_time_split, fully_observed=fully_observed)

        if self.fully_observed:
            self.norm_with_masks(self.base_mask + self.train_mask + self.val_mask + self.test_mask)
        else:
            self.norm_with_masks(self.base_mask + self.train_mask)

    def init_data(self):
        self.x = torch.load(self.processed_paths[0])
        self.edge_index = torch.load(self.processed_paths[1])
        self.node_labels = torch.load(self.processed_paths[2])
        self.coords = torch.load(self.processed_paths[3])
        self.node_ids = torch.load(self.processed_paths[4])

    @property
    def raw_file_names(self):
        return ['edge_outputs_dua_x1_5.xml', 'lust.net.xml']

    @property
    def raw_dir(self):
        return 'data/raw_dir/'

    @property
    def processed_file_names(self):
        return ['data.pt', 'edge_index.pt', 'node_labels.pt', 'coords.pt', 'node_ids.pt']

    @property
    def processed_dir(self) -> str:
        return 'data/processed_dir'

    @property
    def num_node_features(self):
        return 2

    @property
    def num_edge_features(self):
        return 0

    @property
    def num_features(self):
        return self.num_node_features

    @property
    def num_classes(self):
        return 2

    def download(self):
        pass

    def norm_with_masks(self, mask):
        norm_mask = mask.squeeze()
        self.normalize_labels([0, 1])
        self.normalize_features(norm_mask)

    def process(self):
        # process the graph structure and get node_speed dictionary and edge_index tensor for data
        self.process_graph()

        tree = ET.parse(self.raw_paths[0])
        root = tree.getroot()

        # initialize data as a dictionary data[endpoint] = []
        single_timestep_data = {}
        for endpoint, speed in self.nodes.items():
            single_timestep_data[endpoint] = []

        x = torch.zeros(len(root), len(self.nodes), 2)

        for t, child in enumerate(root):

            step_endpoints = np.array(list(self.nodes.keys()))
            for edge in child:
                id = edge.attrib['id']
                if (id not in self.nodes.keys()):
                    continue
                step_endpoints = np.delete(step_endpoints, np.where(step_endpoints == id))
                density = float(edge.attrib.get('density', 0))
                speed = float(edge.attrib.get('speed', 0))
                # occupancy = float(edge.attrib.get('occupancy', 0))
                # traveltime = float(edge.attrib.get('traveltime', 0))
                # timeLoss = float(edge.attrib.get('timeLoss', 0))
                single_timestep_data[id] = [[density, speed]]
            for endpoint in step_endpoints:
                allowed_speed = self.nodes[endpoint][0]
                single_timestep_data[endpoint] = [[0., allowed_speed]]

            for nodeKey in single_timestep_data.keys():
                x[t][self.nodes[nodeKey][-1]] = torch.tensor(single_timestep_data[nodeKey])

        self.x = x
        self.process_labels(self.nodes)
        torch.save(self.x, self.processed_paths[0])
        torch.save(self.edge_index, self.processed_paths[1])
        torch.save(self.node_labels, self.processed_paths[2])
        torch.save(self.coords, self.processed_paths[3])
        torch.save(self.node_ids, self.processed_paths[4])

    def labels_to_one_hot(self, labels):
        categorical = preprocessing.LabelEncoder().fit_transform(labels)
        categorical = torch.LongTensor(categorical).unsqueeze(dim=-1)
        one_hot = torch.FloatTensor(categorical.shape[0], max(categorical) + 1)
        one_hot.zero_()
        one_hot.scatter_(1, categorical, 1)
        return one_hot

    def normalize_features(self, mask):
        masked_features = self.x[:self.train_end(), mask]
        mean, std = masked_features.mean(dim=[0, 1]), masked_features.std(dim=[0, 1])
        self.x = (self.x - mean) / std

    def normalize_labels(self, cols):
        norm_labels = self.node_labels[:, cols]
        mean, std = norm_labels.mean(dim=0), norm_labels.std(dim=0)
        self.node_labels[:, cols] = (self.node_labels[:, cols] - mean) / std

    def process_labels(self, label_dict):
        road_types = []
        priorities = []
        max_speed = []
        road_length = []
        lane_counts = []
        x_coords = []
        y_coords = []
        self.node_ids = []
        idxs = []
        for id, labels in label_dict.items():
            idxs.append(self.node_indices[id])
            self.node_ids.append(id)
            max_speed.append(labels[0])
            x_coords.append(labels[1])
            y_coords.append(labels[2])
            road_length.append(labels[3])
            lane_counts.append(labels[4])
            road_types.append(labels[5])
            priorities.append(labels[6])
        sorted_idxs = np.argsort(idxs)
        max_speed = torch.FloatTensor(max_speed).unsqueeze(dim=-1)[sorted_idxs]
        road_length = torch.FloatTensor(road_length).unsqueeze(dim=-1)[sorted_idxs]
        x_coords = torch.FloatTensor(x_coords)[sorted_idxs]
        y_coords = torch.FloatTensor(y_coords)[sorted_idxs]
        self.coords = torch.stack((x_coords, y_coords), dim=1)

        road_one_hot = self.labels_to_one_hot(road_types)[sorted_idxs]
        priorities_one_hot = self.labels_to_one_hot(priorities)[sorted_idxs]  # 8
        lane_one_hot = self.labels_to_one_hot(lane_counts)[sorted_idxs]  # 5
        self.node_labels = torch.cat([max_speed, road_length, road_one_hot],
                                     dim=1)  # torch.FloatTensor([[labels[0], labels[3]] for id, labels in label_dict.items()])

        # norm_labels = node_labels[norm_mask]
        # mean, std = norm_labels.mean(), norm_labels.std()

    def process_graph(self):
        tree = ET.parse(self.raw_paths[1])
        root = tree.getroot()

        edge_count = 0
        edges = {}

        # node -> index
        node_indices = {}

        # index -> node
        index_nodes = []

        # determines index of next discovered node
        nodeNumber = 0

        # node -> speed
        attributes = {}

        for child in root:
            # speed attribute of nodes
            if child.tag == 'edge' and ('function' not in child.attrib or child.attrib['function'] != 'internal'):
                # num childs
                # type
                # priority
                if child.attrib['id'] in attributes:
                    print('Duplicate!')
                attributes[child.attrib['id']] = [float(child[0].attrib['speed']), ]
                s = child[0].attrib['shape'].split()
                coordinates = [0, 0]
                for att in s:
                    att = att.split(',')
                    coordinates[0] += float(att[0])
                    coordinates[1] += float(att[1])
                coordinates[0] = coordinates[0] / len(s)
                coordinates[1] = coordinates[1] / len(s)
                attributes[child.attrib['id']].append(coordinates[0] / len(s))
                attributes[child.attrib['id']].append(coordinates[1] / len(s))
                attributes[child.attrib['id']].append(float(child[0].attrib['length']))
                attributes[child.attrib['id']].append(len(child))
                if 'type' in child.attrib:
                    attributes[child.attrib['id']].append(child.attrib['type'])
                else:
                    attributes[child.attrib['id']].append('unknown')
                attributes[child.attrib['id']].append(child.attrib['priority'])
            # edge handling
            if child.tag == 'connection':
                if not child.attrib['from'].startswith(':') and not child.attrib['to'].startswith(':'):
                    # number of edges / connections increases by 1
                    edge_count += 2

                    # new edge from
                    if (not (edges.__contains__(child.attrib['from']))):
                        edges[child.attrib['from']] = []
                    if (not (edges.__contains__(child.attrib['to']))):
                        edges[child.attrib['to']] = []
                    edges[child.attrib['from']].append(child.attrib['to'])
                    edges[child.attrib['to']].append(child.attrib['from'])

                    if (not node_indices.__contains__(child.attrib['from'])):
                        node_indices[child.attrib['from']] = nodeNumber
                        index_nodes.append(child.attrib['from'])
                        nodeNumber += 1

                    # to node is new
                    if (not node_indices.__contains__(child.attrib['to'])):
                        node_indices[child.attrib['to']] = nodeNumber
                        index_nodes.append(child.attrib['to'])
                        nodeNumber += 1

                        # get dictionary of unique nodes with speeds from list of all nodes (including duplicates)
        nodes = {endpoint: attributes[endpoint] for endpoint in attributes}

        # reduce amount of nodes to specified amount
        self.nodes, self.node_indices, self.index_nodes = nodes, node_indices, index_nodes

        for nodeKey in self.nodes.keys():
            nodes[nodeKey].append(self.node_indices[nodeKey])

        # get tensor in coo form from the dictionary edges
        self.edge_index = self.compute_edge_index(edge_count, edges, self.node_indices)
        self.edges = edges

    def compute_edge_index(self, edge_count, edges, node_indices):
        edge_index = torch.zeros(2, edge_count, dtype=int)

        i = 0
        for node_key in edges.keys():
            for node_edge in edges[node_key]:
                if (node_key not in node_indices.keys() or node_edge not in node_indices.keys()):
                    continue
                edge_index[0][i] = node_indices[node_key]
                edge_index[1][i] = node_indices[node_edge]
                i += 1

        edge_index = edge_index[:, :i]

        return edge_index
