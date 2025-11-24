"""
Trustee Decision Tree Surrogate for Multiclass Classification
Uses Trustee framework to extract interpretable decision tree from trained models
"""
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz, export_text
from trustee import ClassificationTrustee
import keras
import graphviz

#=============================================================================
# CONFIGURATION
#=============================================================================
PREP_DIR = Path("prep")
WINDOW_SIZE = 29
NUM_FEATURES = 10  # CAN_ID, DLC, DATA[0-7] (Timestamp excluded!)

# Trustee parameters
NUM_ITER = 50
NUM_STABILITY_ITER = 20
SAMPLES_SIZE = 0.5  # Use 50% of data for each iteration
MAX_DEPTH = 12
TOP_K = 20

# Label mapping
LABEL_MAPPING = {
    0: 'Normal',
    1: 'DoS',
    2: 'Fuzzy',
    3: 'Gear_Attack',
    4: 'RPM_Attack'
}

# Feature names
FEATURE_NAMES_BASE = ['CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

#=============================================================================
# DATA LOADING
#=============================================================================
def load_data():
    """Load preprocessed data"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    X_train = np.load(PREP_DIR / "X_train_seq.npy").astype(np.float32)
    y_train = np.load(PREP_DIR / "y_train_seq.npy").astype(np.int8)
    X_test = np.load(PREP_DIR / "X_test_seq.npy").astype(np.float32)
    y_test = np.load(PREP_DIR / "y_test_seq.npy").astype(np.int8)
    
    print(f"  Train: {X_train.shape}, labels: {y_train.shape}")
    print(f"  Test:  {X_test.shape}, labels: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def flatten_sequences(X_3d):
    """Convert (N, window, features) -> (N, window*features) flat"""
    n_samples, window_size, n_features = X_3d.shape
    return X_3d.reshape(n_samples, window_size * n_features)


def unflatten_to_3d(X_flat):
    """Convert (N, window*features) -> (N, window, features)"""
    n_samples = X_flat.shape[0]
    return X_flat.reshape(n_samples, WINDOW_SIZE, NUM_FEATURES)


#=============================================================================
# IDENTIFY NON-CONSTANT COLUMNS
#=============================================================================
def identify_usable_columns(X_train_flat_df):
    """
    Find columns that actually vary (remove constant/padded columns)
    Returns list of column indices to keep
    """
    print("\n" + "="*70)
    print("IDENTIFYING USABLE COLUMNS")
    print("="*70)
    
    full_dim = X_train_flat_df.shape[1]
    
    # Find columns with more than 1 unique value
    non_constant_cols = []
    for i in range(full_dim):
        if X_train_flat_df.iloc[:, i].nunique() > 1:
            non_constant_cols.append(i)
    
    non_constant_cols = sorted(non_constant_cols)
    constant_cols = sorted(set(range(full_dim)) - set(non_constant_cols))
    
    print(f"  Total columns: {full_dim}")
    print(f"  Usable (varying): {len(non_constant_cols)}")
    print(f"  Constant/masked: {len(constant_cols)}")
    
    if len(constant_cols) > 0:
        print(f"  Constant columns will be filled with mean values")
    
    return non_constant_cols, constant_cols


#=============================================================================
# FEATURE RECONSTRUCTION
#=============================================================================
def build_full_from_trimmed(trim_df, col_means, non_constant_cols, full_dim):
    """
    Reconstruct full flat data from trimmed (usable columns only)
    Fill constant columns with their mean values
    """
    n = trim_df.shape[0]
    full = np.tile(col_means, (n, 1))  # Start with means
    full[:, non_constant_cols] = trim_df.values  # Insert usable features
    return full


#=============================================================================
# MODEL WRAPPER FOR TRUSTEE
#=============================================================================
class MulticlassModelWrapper:
    """
    Wrapper for Keras model that Trustee can use
    Takes trimmed flat input, reconstructs full input, predicts
    """
    def __init__(self, keras_model, col_means, non_constant_cols, full_dim):
        self.kmodel = keras_model
        self.col_means = col_means
        self.non_constant_cols = non_constant_cols
        self.full_dim = full_dim
        self.calls = 0
    
    def predict(self, X_trim_flat):
        """
        Trustee calls this method
        X_trim_flat: pandas DataFrame with only usable columns
        Returns: class labels (not probabilities)
        """
        self.calls += 1
        
        # Reconstruct full flat data
        full = build_full_from_trimmed(
            X_trim_flat, 
            self.col_means, 
            self.non_constant_cols, 
            self.full_dim
        )
        
        # Reshape to 3D for model
        X_3d = unflatten_to_3d(full)
        
        # Get predictions
        y_proba = self.kmodel.predict(X_3d, batch_size=2048, verbose=0)
        y_pred = np.argmax(y_proba, axis=1).astype(np.int8)
        
        return y_pred


#=============================================================================
# MAIN PIPELINE
#=============================================================================
def main(model_name='cnn'):
    """
    Extract decision tree from neural network using Trustee
    
    Args:
        model_name: 'cnn', 'lstm', 'ann', or 'transformer'
    """
    print("\n" + "="*70)
    print(f"TRUSTEE MULTICLASS DECISION TREE EXTRACTION")
    print(f"Model: {model_name.upper()}")
    print("="*70)
    
    # ==================== STEP 1: Load Model ====================
    model_path = PREP_DIR / f"{model_name}_final.keras"
    if not model_path.exists():
        model_path = PREP_DIR / f"{model_name}_best.keras"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)
    print("  ✓ Model loaded")
    
    # ==================== STEP 2: Load Data ====================
    X_train_3d, y_train, X_test_3d, y_test = load_data()
    
    # Flatten to 2D for Trustee
    print("\nFlattening sequences...")
    X_train_flat = flatten_sequences(X_train_3d)
    X_test_flat = flatten_sequences(X_test_3d)
    print(f"  Train flat: {X_train_flat.shape}")
    print(f"  Test flat:  {X_test_flat.shape}")
    
    # Convert to DataFrame (Trustee expects pandas)
    X_train_flat_df = pd.DataFrame(X_train_flat)
    X_test_flat_df = pd.DataFrame(X_test_flat)
    
    full_dim = X_train_flat_df.shape[1]
    print(f"  Full dimensionality: {full_dim} ({WINDOW_SIZE} × {NUM_FEATURES})")
    
    # ==================== STEP 3: Identify Usable Columns ====================
    non_constant_cols, constant_cols = identify_usable_columns(X_train_flat_df)
    
    # Compute column means for reconstruction
    col_means = X_train_flat_df.mean(axis=0).values.astype(np.float32)
    
    # ==================== STEP 4: Get Model Predictions ====================
    print("\n" + "="*70)
    print("GETTING MODEL PREDICTIONS")
    print("="*70)
    
    print("  Predicting on train set...")
    y_train_proba = model.predict(X_train_3d, batch_size=2048, verbose=0)
    y_train_hat = np.argmax(y_train_proba, axis=1).astype(np.int8)
    
    print("  Predicting on test set...")
    y_test_proba = model.predict(X_test_3d, batch_size=2048, verbose=0)
    y_test_hat = np.argmax(y_test_proba, axis=1).astype(np.int8)
    
    print("\nModel performance on test set:")
    print(classification_report(
        y_test, y_test_hat,
        target_names=[LABEL_MAPPING[i] for i in range(5)],
        zero_division=0
    ))
    
    # ==================== STEP 5: Build Trimmed Views ====================
    print("\n" + "="*70)
    print("BUILDING TRIMMED VIEWS")
    print("="*70)
    
    X_train_trim = X_train_flat_df.iloc[:, non_constant_cols].copy()
    X_test_trim = X_test_flat_df.iloc[:, non_constant_cols].copy()
    
    print(f"  Train trimmed: {X_train_trim.shape}")
    print(f"  Test trimmed:  {X_test_trim.shape}")
    
    # Feature names for trimmed columns
    def get_feature_name(col_idx):
        timestep = col_idx // NUM_FEATURES
        feature_idx = col_idx % NUM_FEATURES
        return f"t{timestep}_{FEATURE_NAMES_BASE[feature_idx]}"
    
    feature_names = [get_feature_name(i) for i in non_constant_cols]
    print(f"  Feature names: {len(feature_names)} (e.g., {feature_names[:3]}...)")
    
    # ==================== STEP 6: Create Wrapper ====================
    print("\n" + "="*70)
    print("CREATING MODEL WRAPPER")
    print("="*70)
    
    wrapper = MulticlassModelWrapper(
        model, col_means, non_constant_cols, full_dim
    )
    print("  ✓ Wrapper created")
    
    # ==================== STEP 7: Trustee.fit() ====================
    print("\n" + "="*70)
    print("TRAINING TRUSTEE DECISION TREE")
    print("="*70)
    print(f"  Iterations: {NUM_ITER}")
    print(f"  Stability iterations: {NUM_STABILITY_ITER}")
    print(f"  Sample size: {SAMPLES_SIZE}")
    print(f"  Max depth: {MAX_DEPTH}")
    print(f"  Top-k: {TOP_K}")
    
    trustee = ClassificationTrustee(expert=wrapper)
    
    print("\nFitting Trustee (this may take several minutes)...")
    trustee.fit(
        X_train_trim, y_train_hat,
        num_iter=NUM_ITER,
        num_stability_iter=NUM_STABILITY_ITER,
        samples_size=SAMPLES_SIZE,
        max_depth=MAX_DEPTH,
        top_k=TOP_K,
        verbose=True
    )
    
    print(f"\n  Wrapper was called {wrapper.calls} times during training")
    
    # ==================== STEP 8: Explain ====================
    print("\n" + "="*70)
    print("EXTRACTING DECISION TREES")
    print("="*70)
    
    dt, pruned_dt, agreement, fidelity = trustee.explain(top_k=TOP_K)
    
    print(f"\n  Agreement: {agreement:.3f}")
    print(f"  Fidelity:  {fidelity:.3f}")
    print(f"  Unpruned tree nodes: {dt.tree_.node_count}")
    print(f"  Pruned tree nodes:   {pruned_dt.tree_.node_count}")
    
    # ==================== STEP 9: Evaluate Trees ====================
    print("\n" + "="*70)
    print("EVALUATING DECISION TREES")
    print("="*70)
    
    # Unpruned tree
    dt_pred_test = dt.predict(X_test_trim)
    dt_acc = accuracy_score(y_test, dt_pred_test)
    dt_fid = accuracy_score(y_test_hat, dt_pred_test)
    
    # Pruned tree
    pruned_pred_test = pruned_dt.predict(X_test_trim)
    pruned_acc = accuracy_score(y_test, pruned_pred_test)
    pruned_fid = accuracy_score(y_test_hat, pruned_pred_test)
    
    print("\nUnpruned Tree:")
    print(f"  Accuracy (vs ground truth): {dt_acc:.4f}")
    print(f"  Fidelity (vs neural net):   {dt_fid:.4f}")
    
    print("\nPruned Tree:")
    print(f"  Accuracy (vs ground truth): {pruned_acc:.4f}")
    print(f"  Fidelity (vs neural net):   {pruned_fid:.4f}")
    
    print("\nPruned Tree Classification Report (vs ground truth):")
    print(classification_report(
        y_test, pruned_pred_test,
        target_names=[LABEL_MAPPING[i] for i in range(5)],
        zero_division=0
    ))
    
    # ==================== STEP 10: Export Results ====================
    out_dir = PREP_DIR / f"{model_name}_trustee_tree"
    out_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPORTING RESULTS")
    print("="*70)
    print(f"  Output directory: {out_dir}")
    
    class_names = [LABEL_MAPPING[i] for i in range(5)]
    
    # 1. Feature importances
    print("  Exporting feature importances...")
    importances = pruned_dt.feature_importances_
    
    plt.figure(figsize=(12, 6))
    top_indices = np.argsort(importances)[::-1][:20]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.barh(range(len(top_importances)), top_importances)
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Importance')
    plt.title(f'{model_name.upper()} - Pruned Decision Tree Feature Importances (Top 20)')
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_feature_importances.png", dpi=300)
    plt.close()
    
    # Save all importances as CSV
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    imp_df.to_csv(out_dir / f"{model_name}_feature_importances.csv", index=False)
    
    # 2. Export trees as DOT/PDF
    print("  Exporting tree visualizations...")
    
    # Unpruned tree
    dot_unpruned = export_graphviz(
        dt,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=5  # Only show first 5 levels
    )
    with open(out_dir / f"{model_name}_tree_unpruned.dot", "w") as f:
        f.write(dot_unpruned)
    graphviz.Source(dot_unpruned).render(
        out_dir / f"{model_name}_tree_unpruned",
        format="pdf",
        cleanup=True
    )
    
    # Pruned tree
    dot_pruned = export_graphviz(
        pruned_dt,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=5  # Only show first 5 levels
    )
    with open(out_dir / f"{model_name}_tree_pruned.dot", "w") as f:
        f.write(dot_pruned)
    graphviz.Source(dot_pruned).render(
        out_dir / f"{model_name}_tree_pruned",
        format="pdf",
        cleanup=True
    )
    
    # 3. Export tree as text
    print("  Exporting tree text...")
    
    tree_text_unpruned = export_text(dt, feature_names=feature_names)
    with open(out_dir / f"{model_name}_tree_unpruned.txt", "w") as f:
        f.write(tree_text_unpruned)
    
    tree_text_pruned = export_text(pruned_dt, feature_names=feature_names)
    with open(out_dir / f"{model_name}_tree_pruned.txt", "w") as f:
        f.write(tree_text_pruned)
    
    # 4. Save confusion matrices
    print("  Saving confusion matrices...")
    
    cm_pruned = confusion_matrix(y_test, pruned_pred_test)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_pruned, interpolation='nearest', cmap='Blues')
    plt.title(f'{model_name.upper()} - Pruned Tree Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm_pruned.max() / 2.
    for i in range(5):
        for j in range(5):
            plt.text(j, i, format(cm_pruned[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_pruned[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 5. Save summary
    print("  Saving summary...")
    
    summary = {
        'model': model_name,
        'unpruned_nodes': int(dt.tree_.node_count),
        'pruned_nodes': int(pruned_dt.tree_.node_count),
        'unpruned_accuracy': float(dt_acc),
        'unpruned_fidelity': float(dt_fid),
        'pruned_accuracy': float(pruned_acc),
        'pruned_fidelity': float(pruned_fid),
        'agreement': float(agreement),
        'fidelity': float(fidelity)
    }
    
    pd.DataFrame([summary]).to_csv(out_dir / f"{model_name}_summary.csv", index=False)
    
    # 6. Save predictions
    print("  Saving predictions...")
    
    pred_df = pd.DataFrame({
        'y_true': y_test,
        'y_model_pred': y_test_hat,
        'y_pruned_tree_pred': pruned_pred_test
    })
    pred_df.to_csv(out_dir / f"{model_name}_predictions.csv", index=False)
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Model: {model_name.upper()}")
    print(f"  Pruned Tree Complexity: {pruned_dt.tree_.node_count} nodes")
    print(f"  Pruned Tree Accuracy:   {pruned_acc:.4f}")
    print(f"  Pruned Tree Fidelity:   {pruned_fid:.4f}")
    print(f"  Output Directory: {out_dir}")
    print("="*70)
    print("\nFiles created:")
    print(f"  - {model_name}_tree_pruned.pdf       (visual tree)")
    print(f"  - {model_name}_tree_pruned.txt       (text rules)")
    print(f"  - {model_name}_feature_importances.png")
    print(f"  - {model_name}_confusion_matrix.png")
    print(f"  - {model_name}_summary.csv")
    print(f"  - {model_name}_predictions.csv")
    print("\nNext: Use export_tree_for_cpp.py to generate C++ code")
    print("="*70)
    
    return pruned_dt, feature_names, summary


if __name__ == "__main__":
    import sys
    
    # Get model name from command line
    if len(sys.argv) > 1:
        model_name = sys.argv[1].lower()
    else:
        model_name = 'cnn'
    
    if model_name not in ['cnn', 'lstm', 'ann', 'transformer']:
        print(f"Error: Invalid model name '{model_name}'")
        print("Usage: python trustee_multiclass.py [cnn|lstm|ann|transformer]")
        sys.exit(1)
    
    try:
        tree, feature_names, summary = main(model_name)
        print("\n✓ Done! Trustee extraction completed successfully.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
