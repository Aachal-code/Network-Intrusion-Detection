from typing import final
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adadelta
from sklearn.utils import class_weight

from nb15_pre_processing import MAPPING
from utils.dataset import UNSWNB15NodeClassificationDataset
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.data import Data

from utils.model import NodeClassificator
from utils.model import train
from utils.model import predict

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from utils import SEPARATOR
from utils import str2bool
from utils import get_path_of_all
from utils.util import create_directory
from utils.util import plot_confusion_matrix
from utils.util import plot_recall
from utils.util import setup_logger

NAME_DIR: final = '01 - UNSW-NB15'


# ============================================================================
# NEW: SUBSAMPLING FUNCTION
# ============================================================================

def subsample_data(data, target_nodes=1000, method='degree'):
    """
    Subsample large graphs for faster training on CPU
    
    Args:
        data: PyG Data object
        target_nodes: target number of nodes
        method: 'random' or 'degree' (degree keeps most connected nodes)
    
    Returns:
        Subsampled Data object
    """
    num_nodes = data.x.shape[0]
    
    if num_nodes <= target_nodes:
        print(f"   ⚠️  Data has {num_nodes} nodes (< {target_nodes}), no subsampling needed")
        return data
    
    print(f"   📊 Original: {num_nodes} nodes, {data.edge_index.shape[1]} edges")
    
    # Method 1: Keep most connected nodes (better for graphs)
    if method == 'degree':
        degrees = torch.zeros(num_nodes)
        degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))
        sampled_indices = torch.topk(degrees, target_nodes).indices
    
    # Method 2: Random sampling (simpler)
    else:
        sampled_indices = torch.randperm(num_nodes)[:target_nodes]
    
    sampled_indices = sorted(sampled_indices.tolist())
    sampled_indices = torch.tensor(sampled_indices).long()
    
    # Filter edges that connect sampled nodes
    mask = torch.isin(data.edge_index[0], sampled_indices) & \
           torch.isin(data.edge_index[1], sampled_indices)
    subsampled_edges = data.edge_index[:, mask]
    
    # Remap node indices (0 to target_nodes-1)
    node_mapping = {old: new for new, old in enumerate(sampled_indices.tolist())}
    subsampled_edges[0] = torch.tensor([node_mapping[x.item()] for x in subsampled_edges[0]])
    subsampled_edges[1] = torch.tensor([node_mapping[x.item()] for x in subsampled_edges[1]])
    
    # Create new subsampled data
    subsampled_data = Data(
        x=data.x[sampled_indices],
        edge_index=subsampled_edges,
        y=data.y[sampled_indices]
    )
    
    print(f"   ✅ Subsampled: {subsampled_data.x.shape[0]} nodes, {subsampled_data.edge_index.shape[1]} edges")
    print(f"   ⚡ Speed improvement: ~{num_nodes // target_nodes}x faster training")
    
    return subsampled_data


# ============================================================================
# MODIFIED: NODE CLASSIFICATION WITH OPTIMIZATIONS
# ============================================================================

def node_classification(dataset_path: str, binary, n_neigh, augmentation, hid, num_convs, 
                       subsample=True, target_nodes=1000, epochs=30):
    """
    Optimized node classification with subsampling and early stopping
    
    Args:
        dataset_path: path to dataset
        binary: binary classification flag
        n_neigh: number of neighbors
        augmentation: data augmentation flag
        hid: hidden channels
        num_convs: number of convolution layers
        subsample: enable subsampling for CPU training (NEW)
        target_nodes: target number of nodes after subsampling (NEW)
        epochs: number of training epochs (REDUCED from 40)
    """
    
    # Get path of all file
    log_path, log_file_path, log_train_path, model_path, result_path, \
        confusion_matrix_path, detection_rate_path = get_path_of_all(
            NAME_DIR, num_neigh=n_neigh, hid=hid, n_convs=num_convs, augmentation=augmentation)

    for directory in [log_path, model_path, confusion_matrix_path, detection_rate_path]:
        create_directory(directory)

    # Init logger
    logger = setup_logger('logger', log_file_path)
    train_logger = setup_logger('training', log_train_path)

    # Create path of training and test dataset
    _file_name_training = f'UNSW-NB15-train{"-binary" if binary else ""}.csv'
    _file_path_training = os.path.join(dataset_path, 'raw', _file_name_training)
    _file_name_val = f'UNSW-NB15-val{"-binary" if binary else ""}.csv'
    _file_name_testing = f'UNSW-NB15-test{"-binary" if binary else ""}.csv'

    # Create training, val and test dataset
    train_dataset = UNSWNB15NodeClassificationDataset(
        root=dataset_path,
        file_name=_file_name_training,
        binary=binary,
        num_neighbors=n_neigh,
        augmentation=augmentation,
        val=False,
        test=False
    )
    val_dataset = UNSWNB15NodeClassificationDataset(
        root=dataset_path,
        file_name=_file_name_val,
        binary=binary,
        num_neighbors=n_neigh,
        val=True,
        test=False
    )
    test_dataset = UNSWNB15NodeClassificationDataset(
        root=dataset_path,
        file_name=_file_name_testing,
        binary=binary,
        num_neighbors=n_neigh,
        val=False,
        test=True
    )

    logger.info(f'Number of features: {train_dataset.num_features}')
    logger.info(f'Number of classes: {2 if binary else 10}')
    logger.info(SEPARATOR)

    # Define train, val and test loader
    train_data = train_dataset[0]
    val_data = val_dataset[0]
    test_data = test_dataset[0]

    # ========================================================================
    # NEW: SUBSAMPLING FOR CPU OPTIMIZATION
    # ========================================================================
    if subsample:
        print("\n" + "="*70)
        print("🔄 SUBSAMPLING DATA FOR CPU TRAINING OPTIMIZATION")
        print("="*70)
        print(f"   Target: {target_nodes} nodes per dataset")
        print()
        
        print("   Training data:")
        train_data = subsample_data(train_data, target_nodes=target_nodes, method='degree')
        
        print("   Validation data:")
        val_data = subsample_data(val_data, target_nodes=target_nodes, method='degree')
        
        print("   Test data:")
        test_data = subsample_data(test_data, target_nodes=target_nodes, method='degree')
        
        print("\n" + "="*70)
        print("✅ SUBSAMPLING COMPLETE - Ready for fast CPU training!")
        print("="*70 + "\n")
    
    # ========================================================================

    # Log info of train, val and test
    logger.info(train_data)
    logger.info(val_data)
    logger.info(test_data)
    logger.info(SEPARATOR)

    # Define train, val and test loader
    train_loader = RandomNodeLoader(train_data, num_parts=256, shuffle=True)
    val_loader = RandomNodeLoader(val_data, num_parts=256, shuffle=True)
    test_loader = RandomNodeLoader(test_data, num_parts=256, shuffle=True)

    # Define model
    logger.info(f'N. Convs: {num_convs}')
    logger.info(f'N. Hidden Channels: {hid}')
    logger.info(f'Training epochs: {epochs}')  # NEW: Log epochs
    logger.info(f'Subsampling enabled: {subsample}')  # NEW: Log subsampling
    logger.info(SEPARATOR)
    
    train_logger.info(f'N. Convs: {num_convs}')
    train_logger.info(f'N. Hidden Channels: {hid}')
    train_logger.info(f'Training epochs: {epochs}')  # NEW
    train_logger.info(f'Subsampling enabled: {subsample}')  # NEW
    train_logger.info(SEPARATOR)
    
    model = NodeClassificator(
        dataset=train_data,
        num_classes=2 if binary else 10,
        num_convs=num_convs,
        hid=hid,
        alpha=0.5,
        theta=0.7,
        dropout=0.3
    )

    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {str(device).upper()}")
    if device.type == 'cpu':
        print("   💡 CPU mode: Subsampling recommended (already enabled)")
    print()
    
    model.to(device)
    logger.info(model)
    logger.info(SEPARATOR)

    # Define loss function and optimizer
    y = train_dataset[0].y.numpy()
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    y_val = val_dataset[0].y.numpy()
    class_weights_val = class_weight.compute_class_weight('balanced', classes=np.unique(y_val), y=y_val)
    class_weights_val = torch.tensor(class_weights_val, dtype=torch.float)
    criterion_val = nn.CrossEntropyLoss(weight=class_weights_val.to(device))

    optimizer = Adadelta(model.parameters())

    # Check if model is trained
    model_name = f'gc_model_test_attack_{n_neigh}_hid_{hid}_convs_{num_convs}{"_aug" if augmentation else ""}'
    model_trained_path = os.path.join(model_path, f'{model_name}.h5')
    
    if os.path.exists(model_trained_path):
        print(f"✅ Loading pre-trained model: {model_name}")
        model.load_state_dict(torch.load(model_trained_path))
    
    # Train model if is not trained
    else:
        print(f"🚀 Starting training with {epochs} epochs...")
        print(f"   Early stopping enabled (patience=10)")
        print()
        
        train(
            model,
            train_loader,
            criterion,
            criterion_val,
            optimizer,
            device,
            model_path,
            logger=train_logger,
            epochs=epochs,  # NEW: Use optimized epochs parameter
            model_name=model_name,
            evaluation=True,
            val_dataloader=val_loader,
            patience=10  # NEW: Early stopping with patience=10
        )
        train_logger.info(SEPARATOR)

    # Test model
    print("\n📊 Running evaluation on test set...")
    y_true, y_pred = predict(
        model,
        test_loader,
        device
    )

    logger.info(f'Accuracy on test: {accuracy_score(y_true, y_pred)}')
    logger.info(f'Balanced accuracy on test: {balanced_accuracy_score(y_true, y_pred)}')
    logger.info(f'\n{classification_report(y_true, y_pred, digits=3)}')

    print(f"✅ Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"✅ Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}\n")

    # Plot results
    if not binary:
        image_name = f'node-classification_{n_neigh}' \
                     f'_hid_{hid}_convs_{num_convs}' \
                     f'{"_aug" if augmentation else ""}' \
                     f'.png'
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        plot_confusion_matrix(cm, MAPPING.keys(), os.path.join(confusion_matrix_path, image_name))
        # Detection rate
        recall = recall_score(y_true, y_pred, average=None)
        plot_recall(MAPPING.keys(), recall, os.path.join(detection_rate_path, image_name))
    
    print("="*70)
    print("✅ TRAINING AND EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN Network Intrusion Detection (Optimized for CPU)')
    
    # Original arguments
    parser.add_argument('-b', dest='b', action='store',
                        type=str2bool, default=False, help='binary classification (default: False)')
    parser.add_argument('-aug', dest='aug', action='store',
                        type=str2bool, default=True, help='apply augmentation (default: True)')
    parser.add_argument('-neigh', dest='neigh', action='store',
                        type=int, default=1, help='number of neighbors (default: 1)')
    parser.add_argument('-hid', dest='hid', action='store',
                        type=int, default=256, help='hidden channels (default: 256)')
    parser.add_argument('-n_convs', dest='n_convs', action='store',
                        type=int, default=32, help='number of convolution blocks (default: 32)')
    
    # NEW: Optimization arguments
    parser.add_argument('-subsample', dest='subsample', action='store',
                        type=str2bool, default=True, help='enable subsampling (default: True)')
    parser.add_argument('-target_nodes', dest='target_nodes', action='store',
                        type=int, default=1000, help='target nodes after subsampling (default: 1000)')
    parser.add_argument('-epochs', dest='epochs', action='store',
                        type=int, default=30, help='number of training epochs (default: 30)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 GNN NETWORK INTRUSION DETECTION (OPTIMIZED FOR CPU)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Binary mode: {args.b}")
    print(f"  Augmentation: {args.aug}")
    print(f"  Neighbors: {args.neigh}")
    print(f"  Hidden channels: {args.hid}")
    print(f"  Convolution layers: {args.n_convs}")
    print(f"  Subsampling: {args.subsample}")
    print(f"  Target nodes: {args.target_nodes}")
    print(f"  Training epochs: {args.epochs}")
    print("="*70 + "\n")

    node_classification(
        os.path.join(os.getcwd(), 'dataset'), 
        args.b, 
        args.neigh, 
        args.aug, 
        args.hid, 
        args.n_convs,
        subsample=args.subsample,  # NEW
        target_nodes=args.target_nodes,  # NEW
        epochs=args.epochs  # NEW
    )