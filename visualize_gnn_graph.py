"""
GNN Network Intrusion Detection - Graph Visualization
Visualizes the learned graph with model predictions as node colors
"""

import torch
import networkx as nx
import numpy as np
from pyvis.network import Network
import os
import sys

# Adjust paths as needed for your setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from torch_geometric.data import Data
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths
DATASET_PATH = r'C:\Users\aacha\OneDrive\Desktop\graph\network-intrusion-detection-gnn\dataset'
MODEL_PATH = r'C:\Users\aacha\OneDrive\Desktop\graph\network-intrusion-detection-gnn\checkpoints\best_model.pt'

# Class names for UNSW-NB15
CLASS_NAMES = {
    0: 'Normal',
    1: 'Fuzzers',
    2: 'Analysis',
    3: 'Backdoor',
    4: 'DoS',
    5: 'Exploits',
    6: 'Generic',
    7: 'Reconnaissance',
    8: 'Shellcode',
    9: 'Worms'
}

# Color mapping for classes
COLOR_MAP = {
    0: '#00FF00',    # Green = Normal
    1: '#FF6B6B',    # Red = Fuzzers
    2: '#FFA500',    # Orange = Analysis
    3: '#FF1493',    # Deep Pink = Backdoor
    4: '#DC143C',    # Crimson = DoS
    5: '#8B0000',    # Dark Red = Exploits
    6: '#FFD700',    # Gold = Generic
    7: '#FF4500',    # Orange Red = Reconnaissance
    8: '#9932CC',    # Dark Orchid = Shellcode
    9: '#FF00FF'     # Magenta = Worms
}

# ============================================================================
# LOAD PROCESSED DATA
# ============================================================================

def load_processed_data(dataset_path):
    """Load processed graph data safely"""

    processed_dir = os.path.join(dataset_path, 'processed')

    train_file = os.path.join(processed_dir, 'nb15_1_aug.pt')
    val_file   = os.path.join(processed_dir, 'nb15_val_1.pt')
    test_file  = os.path.join(processed_dir, 'nb15_test_1.pt')

    try:
        # FIX: handle tuple OR direct Data object safely
        def safe_load(path):
            obj = torch.load(path, weights_only=False)
            return obj[0] if isinstance(obj, (tuple, list)) else obj

        train_data = safe_load(train_file)
        val_data   = safe_load(val_file)
        test_data  = safe_load(test_file)

        print("✅ Loaded train, validation and test graphs")

        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def load_predictions(model, data, device='cpu'):
    """
    Run model on data to get predictions
    Requires your trained model to be loaded
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()
    return predictions


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_networkx_graph(edge_index, predictions, node_ids=None, max_nodes=1000):
    """
    Convert PyG graph to NetworkX graph
    
    Args:
        edge_index: torch tensor [2, num_edges]
        predictions: numpy array of predicted classes
        node_ids: optional custom node IDs
        max_nodes: subsample for visualization clarity
    """
    
    num_nodes = predictions.shape[0]
    print(f"📊 Total nodes: {num_nodes}")
    print(f"   Total edges: {edge_index.shape[1]}")
    
 # Subsample if too large
    if num_nodes > max_nodes:
        print(f" Graph too large. Subsampling to {max_nodes} nodes...")

    # Compute degree
    degrees = torch.zeros(num_nodes)
    degrees.scatter_add_(
        0,
        edge_index[0],
        torch.ones(edge_index.shape[1])
    )

    # Pick top nodes
    top_nodes = torch.topk(degrees, max_nodes).indices
    top_nodes_set = set(top_nodes.tolist())

    # Map old -> new indices
    node_mapping = {old.item(): new for new, old in enumerate(top_nodes)}

    # Filter edges safely
    src_list = []
    dst_list = []

    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if s in top_nodes_set and d in top_nodes_set:
            src_list.append(node_mapping[s])
            dst_list.append(node_mapping[d])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Fix predictions
    predictions = predictions[top_nodes.cpu().numpy()]
    num_nodes = max_nodes
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node_id in range(num_nodes):
        pred_class = int(predictions[node_id])
        G.add_node(node_id, 
                   label=f"N{node_id}",
                   class_label=CLASS_NAMES.get(pred_class, 'Unknown'),
                   prediction=pred_class)
    
    # Add edges
    for src, dst in zip(edge_index[0].numpy(), edge_index[1].numpy()):
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            G.add_edge(int(src), int(dst))
    
    print(f"✅ NetworkX graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ============================================================================
# VISUALIZE WITH PYVIS
# ============================================================================

def visualize_graph(G, predictions, output_file='intrusion_detection_graph.html'):
    """
    Create interactive PyVis visualization
    """
    # Create PyVis network
    net = Network(
        height='750px',
        width='100%',
        directed=False,
        notebook=False
)

    net.toggle_physics(True)
    
    # Configure physics
    net.show_buttons(filter_=['physics'])
    
    # Count class distribution for legend
    class_counts = defaultdict(int)
    for node in G.nodes():
        pred_class = G.nodes[node]['prediction']
        class_counts[pred_class] += 1
    
    # Add nodes with colors
    for node in G.nodes():
        pred_class = G.nodes[node]['prediction']
        class_name = CLASS_NAMES.get(pred_class, 'Unknown')
        color = COLOR_MAP.get(pred_class, '#CCCCCC')
        
        # Size by node degree (connectivity)
        degree = G.degree(node)
        size = max(15, min(50, degree * 2))
        
        net.add_node(
            node,
            label=f"{class_name}\n(Node {node})",
            title=f"{class_name} - Degree: {degree}",
            color=color,
            size=size
        )
    
    # Add edges
    for src, dst in G.edges():
        net.add_edge(src, dst, color='#999999', width=0.5)
    
    # Save and show
    net.write_html(output_file, open_browser=True)
    print(f"\n✅ Graph visualization saved to: {output_file}")
    print(f"📂 Open this file in a web browser to interact with the graph")
    
    # Print summary
    print(f"\n📊 Class Distribution in Graph:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / len(predictions)) * 100
        print(f"   {CLASS_NAMES.get(class_id, 'Unknown'):20s}: {count:6d} nodes ({percentage:5.2f}%)")


# ============================================================================
# STATISTICS
# ============================================================================

def print_graph_stats(G, predictions):
    """Print graph statistics"""
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    
    print(f"\nNodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"\nNode Degree Stats:")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Mean degree: {np.mean(degrees):.2f}")
    
    # Connected components
    if isinstance(G, nx.DiGraph):
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    print(f"\nConnected components: {len(components)}")
    print(f"  Largest component size: {len(max(components, key=len))}")
    
    # Class balance
    print(f"\nClass Distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for class_id, count in zip(unique, counts):
        pct = (count / len(predictions)) * 100
        print(f"  {CLASS_NAMES.get(class_id, 'Unknown'):20s}: {count:6d} ({pct:5.2f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("GNN NETWORK INTRUSION DETECTION - GRAPH VISUALIZATION")
    print("="*60)
    
    # Load data
    print("\n1️⃣ Loading processed data...")
    data_list = load_processed_data(DATASET_PATH)
    if data_list is None:
        return
    
    # Use test data (last in list)
    train_data = data_list['train']
    val_data = data_list['val']
    test_data = data_list['test']
    print(f"   Using test set: {test_data}")
    
    # For now, create dummy predictions if model not available
    # In real scenario, you'd load trained model and run inference
    print("\n2️⃣ Getting predictions...")
    print(f"   ⚠️  Using random predictions for demo")
    print(f"   (In production, load your trained model and call model(data.x, data.edge_index))")
    
    # Dummy predictions for visualization demo
    predictions = np.random.randint(0, 10, test_data.x.shape[0])
    
    # Build graph
    print("\n3️⃣ Building NetworkX graph...")
    G = build_networkx_graph(test_data.edge_index, predictions, max_nodes=500)
    
    # Print stats
    print_graph_stats(G, predictions)
    
    # Visualize
    print("\n4️⃣ Creating interactive visualization...")
    visualize_graph(G, predictions)
    
    print("\n" + "="*60)
    print("✅ Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()