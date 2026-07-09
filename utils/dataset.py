from typing import Union, List, Tuple
from collections import Counter

import os
import pandas as pd
import numpy as np
import torch

from torch_geometric.data import InMemoryDataset, Data


class UNSWNB15NodeClassificationDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        file_name,
        num_neighbors=2,
        binary: bool = False,
        augmentation: bool = False,
        val: bool = False,
        test: bool = False,
        transform=None,
        pre_transform=None
    ):

        self.file_name = file_name
        self.num_neighbors = num_neighbors
        self.binary = binary
        self.augmentation = augmentation
        self.val = val
        self.test = test

        super().__init__(root, transform, pre_transform)

        # --------------------------
        # SAFE LOAD (PyTorch 2.6 FIX)
        # --------------------------
        if os.path.exists(self.processed_paths[0]):
            import torch.serialization
            import torch_geometric.data.data

            torch.serialization.add_safe_globals([
                torch_geometric.data.data.DataEdgeAttr
            ])

            self.data, self.slices = torch.load(
                self.processed_paths[0],
                weights_only=False
            )

    # --------------------------
    # FILES
    # --------------------------
    @property
    def processed_file_names(self):
        if self.val:
            return [f'nb15_val_{"binary_" if self.binary else ""}{self.num_neighbors}.pt']
        elif self.test:
            return [f'nb15_test_{"binary_" if self.binary else ""}{self.num_neighbors}.pt']
        else:
            return [f'nb15_{"binary_" if self.binary else ""}{self.num_neighbors}'
                    f'{"_aug" if self.augmentation else ""}.pt']

    @property
    def raw_file_names(self):
        return self.file_name

    def download(self):
        pass

    # --------------------------
    # PROCESS
    # --------------------------
    def process(self):

        print("RAW PATHS:", self.raw_paths)

        df = pd.read_csv(self.raw_paths[0])
        print("Original shape:", df.shape)

        # --------------------------
        # FEATURES
        # --------------------------
        extract_col = [
          'sload', 'dload', 'smeansz', 'tcprtt', 'sintpkt',
    'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst',
    'proto_tcp', 'state_con' ]

        df_features = df[extract_col]
        y = df['label'].values

        print("Class distribution:", Counter(y))

        # --------------------------
        # NODE FEATURES
        # --------------------------
        x = torch.tensor(df_features.values, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)

        # --------------------------
        # EDGE BUILDING (OPTIMIZED)
        # --------------------------
        edge_index = self.build_edges_fast(df)

        data = Data(x=x, edge_index=edge_index, y=y)

        # --------------------------
        # SAFE SAVE (PyG STANDARD)
        # --------------------------
        self.data, self.slices = self.collate([data])
        torch.save((self._data, self.slices), self.processed_paths[0])

    # --------------------------
    # FAST EDGE BUILDER
    # --------------------------
    def build_edges_fast(self, df):

        features_to_link = [
            'proto_tcp', 'proto_udp', 'proto_other',
            'state_fin', 'state_con', 'state_int', 'state_other',
            'service_-', 'service_dns', 'service_other'
        ]

        row, col = [], []
        MAX_GROUP_SIZE = 1000

        for feat in features_to_link:

            if feat not in df.columns:
                continue

            groups = df.groupby(feat).indices

            for _, idx in groups.items():

                idx = np.asarray(idx)

                if len(idx) < 2:
                    continue

                if len(idx) > MAX_GROUP_SIZE:
                    idx = idx[:MAX_GROUP_SIZE]

                # chain edges (fast, stable)
                src = idx[:-1]
                dst = idx[1:]

                row.extend(src)
                col.extend(dst)

                # undirected edges
                row.extend(dst)
                col.extend(src)

        edge_index = torch.tensor([row, col], dtype=torch.long)

        print("Edge index shape:", edge_index.shape)

        return edge_index

    # --------------------------
    # REQUIRED
    # --------------------------
    def len(self):
        return 1

    def get(self, idx: int):
        return self._data