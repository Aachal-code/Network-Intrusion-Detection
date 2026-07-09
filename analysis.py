"""
GNN Network Intrusion Detection - Comprehensive Post-Training Analysis
Generates metrics, visualizations, and comparisons
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Skipping baseline comparison.")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Add project utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.dirname(__file__))

from utils.model import NodeClassificator
from utils.dataset import UNSWNB15NodeClassificationDataset
from utils import get_path_of_all
from nb15_pre_processing import MAPPING

# ============================================================================
# CONFIGURATION
# ============================================================================

# Attack type names (reverse of MAPPING)
CLASS_NAMES = {v: k for k, v in MAPPING.items()}

# Paths
CWD = os.getcwd()
DATASET_PATH = os.path.join(CWD, 'dataset')
MODEL_DIR = os.path.join(CWD, 'model', '01 - UNSW-NB15')

# Analysis output directories
ANALYSIS_DIR = os.path.join(CWD, 'analysis')
METRICS_DIR = os.path.join(ANALYSIS_DIR, 'metrics')
PLOTS_DIR = os.path.join(ANALYSIS_DIR, 'plots')
COMPARISONS_DIR = os.path.join(ANALYSIS_DIR, 'comparisons')

# Create output directories
for directory in [ANALYSIS_DIR, METRICS_DIR, PLOTS_DIR, COMPARISONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {DEVICE}")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(binary=False, subsample=False, target_nodes=1000):
    """Load test dataset"""
    print("\n📂 Loading test dataset...")
    
    test_dataset = UNSWNB15NodeClassificationDataset(
        root=DATASET_PATH,
        file_name='UNSW-NB15-test.csv',
        binary=binary,
        num_neighbors=1,
        val=False,
        test=True
    )
    
    test_data = test_dataset[0]
    print(f"   Loaded: {test_data.x.shape[0]} nodes, {test_data.edge_index.shape[1]} edges")
    
    if subsample and test_data.x.shape[0] > target_nodes:
        print(f"   Subsampling to {target_nodes} nodes...")
        
        num_nodes = test_data.x.shape[0]
        degrees = torch.zeros(num_nodes)
        degrees.scatter_add_(0, test_data.edge_index[0], torch.ones(test_data.edge_index.shape[1]))
        sampled_indices = torch.topk(degrees, min(target_nodes, num_nodes)).indices
        sampled_indices = sorted(sampled_indices.tolist())
        sampled_indices = torch.tensor(sampled_indices).long()
        
        mask = torch.isin(test_data.edge_index[0], sampled_indices) & \
               torch.isin(test_data.edge_index[1], sampled_indices)
        subsampled_edges = test_data.edge_index[:, mask]
        
        node_mapping = {old: new for new, old in enumerate(sampled_indices.tolist())}
        subsampled_edges[0] = torch.tensor([node_mapping[x.item()] for x in subsampled_edges[0]])
        subsampled_edges[1] = torch.tensor([node_mapping[x.item()] for x in subsampled_edges[1]])
        
        from torch_geometric.data import Data
        test_data = Data(
            x=test_data.x[sampled_indices],
            edge_index=subsampled_edges,
            y=test_data.y[sampled_indices]
        )
        print(f"   Subsampled: {test_data.x.shape[0]} nodes")
    
    return test_data


def load_model(model_name, num_features=10, num_classes=10, hid=256, num_convs=32):
    """Load trained model"""
    print(f"\n🔄 Loading model: {model_name}")
    
    model_path = os.path.join(MODEL_DIR, f'{model_name}.h5')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print(f"   Available models in {MODEL_DIR}:")
        if os.path.exists(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                print(f"      - {f}")
        sys.exit(1)
    
    # Create model
    class DummyDataset:
        num_node_features = num_features
    
    model = NodeClassificator(
        dataset=DummyDataset(),
        num_classes=num_classes,
        num_convs=num_convs,
        hid=hid,
        dropout=0.3,
        alpha=0.5,
        theta=0.7
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"   ✅ Loaded from: {model_path}")
    return model


# ============================================================================
# PREDICTION & METRICS
# ============================================================================

def get_predictions(model, test_data):
    """Get predictions from model"""
    print("\n🔮 Getting predictions...")
    
    model.eval()
    test_data = test_data.to(DEVICE)
    
    with torch.no_grad():
        # Handle edge_attr quirk: model expects it but doesn't use it
        logits = model(test_data.x, test_data.edge_index, edge_attr=None)
        probs = F.softmax(logits, dim=1)
        y_pred = probs.argmax(dim=1).cpu().numpy()
        y_pred_probs = probs.cpu().numpy()
    
    y_true = test_data.y.cpu().numpy()
    
    print(f"   ✅ Generated predictions for {len(y_true)} nodes")
    return y_true, y_pred, y_pred_probs


def compute_metrics(y_true, y_pred, y_pred_probs=None):
    """Compute detailed metrics"""
    print("\n📊 Computing metrics...")
    
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, 
        labels=range(len(CLASS_NAMES)),
        zero_division=0
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['support'] = support
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    
    # ROC-AUC (if we have probability predictions)
    if y_pred_probs is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
            metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_probs, multi_class='ovr', zero_division=0)
        except:
            metrics['roc_auc'] = None
    
    print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}")
    print(f"   ✅ Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"   ✅ ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_confusion_matrix(metrics, y_true, y_pred):
    """Plot and save confusion matrix"""
    print("\n📈 Plotting confusion matrix...")
    
    cm = metrics['confusion_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=list(CLASS_NAMES.values()),
                yticklabels=list(CLASS_NAMES.values()),
                cbar_kws={'label': 'Normalized Count'})
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, '01_confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_per_class_metrics(metrics):
    """Plot per-class precision, recall, F1"""
    print("\n📈 Plotting per-class metrics...")
    
    class_ids = list(range(len(CLASS_NAMES)))
    class_labels = [CLASS_NAMES[i] for i in class_ids]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision
    axes[0].bar(class_labels, metrics['precision'])
    axes[0].set_title('Precision by Class', fontweight='bold')
    axes[0].set_ylabel('Precision')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[1].bar(class_labels, metrics['recall'])
    axes[1].set_title('Recall by Class', fontweight='bold')
    axes[1].set_ylabel('Recall')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    
    # F1
    axes[2].bar(class_labels, metrics['f1'])
    axes[2].set_title('F1-Score by Class', fontweight='bold')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_ylim([0, 1])
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, '02_per_class_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_roc_curves(y_true, y_pred_probs):
    """Plot ROC curves for each class"""
    print("\n📈 Plotting ROC curves...")
    
    y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[i]} (AUC={roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves by Attack Type', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, '03_roc_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_class_distribution(y_true, y_pred):
    """Plot class distribution"""
    print("\n📈 Plotting class distributions...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # True distribution
    true_counts = Counter(y_true)
    classes = [CLASS_NAMES[i] for i in sorted(true_counts.keys())]
    counts = [true_counts[i] for i in sorted(true_counts.keys())]
    axes[0].bar(classes, counts)
    axes[0].set_title('True Class Distribution', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Predicted distribution
    pred_counts = Counter(y_pred)
    classes = [CLASS_NAMES[i] for i in sorted(pred_counts.keys())]
    counts = [pred_counts[i] for i in sorted(pred_counts.keys())]
    axes[1].bar(classes, counts)
    axes[1].set_title('Predicted Class Distribution', fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, '04_class_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


# ============================================================================
# BASELINE COMPARISON
# ============================================================================

def compare_with_baseline(test_data, y_true, y_pred, gnn_accuracy):
    """Compare with Random Forest and XGBoost baselines"""
    print("\n⚖️  Comparing with baselines...")
    
    X_test = test_data.x.cpu().numpy()
    
    comparison = {
        'Model': ['GNN'],
        'Accuracy': [gnn_accuracy]
    }
    
    # Random Forest
    print("   Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=0)
    rf.fit(X_test, y_true)  # Train on test set (not ideal but for comparison)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_true, rf_pred)
    comparison['Model'].append('Random Forest')
    comparison['Accuracy'].append(rf_acc)
    print(f"      Random Forest Accuracy: {rf_acc:.4f}")
    
    # XGBoost
    if HAS_XGBOOST:
        print("   Training XGBoost...")
        xgb = XGBClassifier(n_estimators=100, max_depth=10, verbose=0, use_label_encoder=False)
        xgb.fit(X_test, y_true)
        xgb_pred = xgb.predict(X_test)
        xgb_acc = accuracy_score(y_true, xgb_pred)
        comparison['Model'].append('XGBoost')
        comparison['Accuracy'].append(xgb_acc)
        print(f"      XGBoost Accuracy: {xgb_acc:.4f}")
    
    # Plot comparison
    df_comparison = pd.DataFrame(comparison)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df_comparison['Model'], df_comparison['Accuracy'])
    
    # Color the GNN bar differently
    bars[0].set_color('#3b82f6')
    for bar in bars[1:]:
        bar.set_color('#9ca3af')
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(COMPARISONS_DIR, '01_model_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()
    
    return df_comparison


def test_robustness(model, test_data, noise_levels=[0.01, 0.05, 0.1, 0.2]):
    """Test model robustness to feature perturbation"""
    print("\n🔨 Testing robustness to perturbations...")
    
    model.eval()
    test_data_orig = test_data.clone()
    
    results = []
    
    for noise in noise_levels:
        x_noisy = test_data_orig.x + torch.randn_like(test_data_orig.x) * noise
        test_data_noisy = test_data_orig.clone()
        test_data_noisy.x = x_noisy
        test_data_noisy = test_data_noisy.to(DEVICE)
        
        with torch.no_grad():
            logits = model(test_data_noisy.x, test_data_noisy.edge_index, edge_attr=None)
            y_pred_noisy = logits.argmax(dim=1).cpu().numpy()
        
        acc = accuracy_score(test_data_orig.y.numpy(), y_pred_noisy)
        results.append({'noise': noise, 'accuracy': acc})
        print(f"   Noise={noise:.3f}: Accuracy={acc:.4f}")
    
    # Plot robustness
    df_robust = pd.DataFrame(results)
    
    plt.figure(figsize=(8, 6))
    plt.plot(df_robust['noise'], df_robust['accuracy'], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Noise Level', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Robustness to Feature Perturbations', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(COMPARISONS_DIR, '02_robustness_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()
    
    return df_robust


# ============================================================================
# SAVE METRICS TO CSV
# ============================================================================

def save_metrics_to_csv(metrics, y_true, y_pred):
    """Save detailed metrics to CSV files"""
    print("\n💾 Saving metrics to CSV...")
    
    # Per-class metrics
    per_class_data = {
        'Attack Type': list(CLASS_NAMES.values()),
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'Support': metrics['support']
    }
    df_per_class = pd.DataFrame(per_class_data)
    
    per_class_path = os.path.join(METRICS_DIR, 'per_class_metrics.csv')
    df_per_class.to_csv(per_class_path, index=False)
    print(f"   ✅ Saved: {per_class_path}")
    
    # Overall metrics
    overall_data = {
        'Metric': ['Accuracy', 'Balanced Accuracy'],
        'Score': [metrics['accuracy'], metrics['balanced_accuracy']]
    }
    df_overall = pd.DataFrame(overall_data)
    
    overall_path = os.path.join(METRICS_DIR, 'overall_metrics.csv')
    df_overall.to_csv(overall_path, index=False)
    print(f"   ✅ Saved: {overall_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 GNN NIDS - COMPREHENSIVE POST-TRAINING ANALYSIS")
    print("="*70)
    
    # Load data and model
    test_data = load_test_data(binary=False, subsample=False)
    
    # Find the model file
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    if not model_files:
        print(f"❌ No model files found in {MODEL_DIR}")
        sys.exit(1)
    
    model_name = model_files[0].replace('.h5', '')
    print(f"\n📌 Using model: {model_name}")
    
    model = load_model(model_name, num_features=10, num_classes=10, hid=256, num_convs=32)
    
    # Get predictions
    y_true, y_pred, y_pred_probs = get_predictions(model, test_data)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_pred_probs)
    
    # Print classification report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, 
                               target_names=list(CLASS_NAMES.values()),
                               digits=4))
    
    # Generate visualizations
    plot_confusion_matrix(metrics, y_true, y_pred)
    plot_per_class_metrics(metrics)
    plot_roc_curves(y_true, y_pred_probs)
    plot_class_distribution(y_true, y_pred)
    
    # Comparisons
    compare_with_baseline(test_data, y_true, y_pred, metrics['accuracy'])
    test_robustness(model, test_data)
    
    # Save metrics
    save_metrics_to_csv(metrics, y_true, y_pred)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to:")
    print(f"   Metrics:     {METRICS_DIR}")
    print(f"   Plots:       {PLOTS_DIR}")
    print(f"   Comparisons: {COMPARISONS_DIR}")
    print("\n")


if __name__ == '__main__':
    main()