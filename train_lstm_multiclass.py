"""
LSTM Autoencoder for CAN Bus Anomaly Detection
Trains ONLY on Normal data, detects anomalies by reconstruction error
"""
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import keras
from keras import layers, Input, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
import torch
torch.manual_seed(42)

WINDOW_SIZE = 29


def load_preprocessed_data():
    """Load preprocessed data for anomaly detection"""
    print("="*60)
    print("LOADING PREPROCESSED ANOMALY DETECTION DATA")
    print("="*60)
    
    prep_dir = Path("prep")
    
    required_files = [
        "X_train_anomaly.npy", "y_train_anomaly.npy",
        "X_test_anomaly.npy", "y_test_anomaly.npy",
        "X_val_anomaly.npy", "y_val_anomaly.npy",
        "label_mapping_anomaly.pkl"
    ]
    
    missing = [f for f in required_files if not (prep_dir / f).exists()]
    if missing:
        print(f"❌ Missing files: {missing}")
        print("Please run 'python prepare_canbus_anomaly.py' first.")
        return None, None, None, None, None, None, None
    
    print("Loading data...")
    X_train = np.load(prep_dir / "X_train_anomaly.npy")
    y_train = np.load(prep_dir / "y_train_anomaly.npy")
    X_test = np.load(prep_dir / "X_test_anomaly.npy")
    y_test = np.load(prep_dir / "y_test_anomaly.npy")
    X_val = np.load(prep_dir / "X_val_anomaly.npy")
    y_val = np.load(prep_dir / "y_val_anomaly.npy")
    
    with open(prep_dir / "label_mapping_anomaly.pkl", "rb") as f:
        label_mapping = pickle.load(f)
    
    print(f"✓ Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Normal: {(y_train == 0).sum():,}, Anomaly: {(y_train == 1).sum():,}")
    print(f"✓ Test:  X={X_test.shape}, y={y_test.shape}")
    print(f"  Normal: {(y_test == 0).sum():,}, Anomaly: {(y_test == 1).sum():,}")
    print(f"✓ Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Normal: {(y_val == 0).sum():,}, Anomaly: {(y_val == 1).sum():,}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, label_mapping


def scale_features(X_train, X_test, X_val):
    """Scale features using StandardScaler"""
    print("\nScaling features...")
    scaler = StandardScaler()
    
    n_train, window_size, n_features = X_train.shape
    n_test, n_val = X_test.shape[0], X_val.shape[0]
    
    # Flatten for scaling
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    
    # Fit on training data ONLY
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, window_size, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_test, window_size, n_features)
    X_val_scaled = scaler.transform(X_val_flat).reshape(n_val, window_size, n_features)
    
    # Save scaler
    prep_dir = Path("prep")
    with open(prep_dir / "lstm_anomaly_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved")
    
    return X_train_scaled, X_test_scaled, X_val_scaled, scaler


def build_lstm_autoencoder(input_shape):
    """
    Build LSTM Autoencoder for anomaly detection
    Architecture: LSTM Encoder -> Latent Space -> LSTM Decoder
    """
    print("\n" + "="*60)
    print("BUILDING LSTM AUTOENCODER")
    print("="*60)
    print(f"Input shape: {input_shape}")
    print("Training on: Normal data ONLY")
    print("Detection method: Reconstruction error")
    
    inputs = Input(shape=input_shape)
    
    # ENCODER: Compress the sequence
    x = layers.LSTM(128, activation='relu', return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    
    x = layers.LSTM(64, activation='relu', return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    
    # Latent representation
    encoded = layers.LSTM(32, activation='relu', return_sequences=False)(x)
    
    # DECODER: Reconstruct the sequence
    # Repeat latent vector to match sequence length
    x = layers.RepeatVector(input_shape[0])(encoded)
    
    x = layers.LSTM(32, activation='relu', return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.LSTM(64, activation='relu', return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.LSTM(128, activation='relu', return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer: Reconstruct original input
    decoded = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)
    
    model = Model(inputs, decoded)
    
    # Compile with MSE loss (reconstruction error)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    return model


def train_autoencoder(model, X_train, X_val, y_val):
    """
    Train autoencoder on Normal data ONLY
    Validates on both Normal and Anomaly data
    """
    print("\n" + "="*60)
    print("TRAINING LSTM AUTOENCODER")
    print("="*60)
    print("Training on: Normal data ONLY")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,} (Normal + Anomaly)")
    
    prep_dir = Path("prep")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            str(prep_dir / "lstm_anomaly_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Use ONLY Normal validation samples for early stopping to avoid anomaly loss spikes
    normal_mask = y_val == 0
    if normal_mask.sum() == 0:
        print("  ⚠️  No Normal samples in validation set! Using full val set for monitoring.")
        val_inputs = X_val
    else:
        val_inputs = X_val[normal_mask]
        print(f"  ✓ Using {len(val_inputs):,} Normal samples for validation monitoring")

    # Train autoencoder to reconstruct Normal data
    history = model.fit(
        X_train, X_train,  # Input and target are the same!
        validation_data=(val_inputs, val_inputs),
        epochs=100,
        batch_size=256,
        callbacks=callbacks,
        shuffle=False,  # Preserve temporal order
        verbose=1
    )
    
    return history


def find_threshold(model, X_val, y_val):
    """
    Find optimal threshold for anomaly detection using validation set
    """
    print("\n" + "="*60)
    print("FINDING OPTIMAL THRESHOLD")
    print("="*60)
    
    # Calculate reconstruction errors
    X_val_pred = model.predict(X_val, batch_size=256, verbose=0)
    mse = np.mean(np.square(X_val - X_val_pred), axis=(1, 2))
    
    # Use ROC curve to find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_val, mse)
    
    # Find threshold that maximizes (TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"✓ Optimal threshold: {optimal_threshold:.6f}")
    print(f"  TPR: {tpr[optimal_idx]:.4f}")
    print(f"  FPR: {fpr[optimal_idx]:.4f}")
    
    return optimal_threshold, mse


def evaluate_anomaly_detection(model, X_test, y_test, threshold):
    """
    Evaluate anomaly detection performance
    """
    print("\n" + "="*60)
    print("EVALUATING ANOMALY DETECTION")
    print("="*60)
    
    # Calculate reconstruction errors
    X_test_pred = model.predict(X_test, batch_size=256, verbose=0)
    mse = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))
    
    # Classify: High reconstruction error = Anomaly
    y_pred = (mse > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, mse)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Anomaly'],
        zero_division=0
    ))
    
    # Confusion matrix
    prep_dir = Path("prep")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Normal', 'Anomaly'], columns=['Normal', 'Anomaly'])
    cm_df.to_csv(prep_dir / "lstm_anomaly_confusion_matrix.csv")
    print(f"\n✓ Confusion matrix saved")
    
    # Save predictions
    results_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Reconstruction_Error': mse,
        'Threshold': threshold,
        'Correct': y_test == y_pred
    })
    results_df.to_csv(prep_dir / "lstm_anomaly_predictions.csv", index=False)
    print(f"✓ Predictions saved")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    for label, name in [(0, 'Normal'), (1, 'Anomaly')]:
        mask = y_test == label
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  {name}: {class_acc:.4f} ({mask.sum():,} samples)")
    
    # Save threshold
    with open(prep_dir / "lstm_anomaly_threshold.pkl", "wb") as f:
        pickle.dump(threshold, f)
    print(f"\n✓ Threshold saved: {threshold:.6f}")
    
    return accuracy, y_pred, mse


def plot_reconstruction_error_distribution(mse_val, y_val, mse_test, y_test, threshold):
    """Plot reconstruction error distributions"""
    prep_dir = Path("prep")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Validation set
    ax1.hist(mse_val[y_val == 0], bins=50, alpha=0.5, label='Normal', color='blue')
    ax1.hist(mse_val[y_val == 1], bins=50, alpha=0.5, label='Anomaly', color='red')
    ax1.axvline(threshold, color='black', linestyle='--', label=f'Threshold={threshold:.4f}')
    ax1.set_xlabel('Reconstruction Error (MSE)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Validation Set')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Test set
    ax2.hist(mse_test[y_test == 0], bins=50, alpha=0.5, label='Normal', color='blue')
    ax2.hist(mse_test[y_test == 1], bins=50, alpha=0.5, label='Anomaly', color='red')
    ax2.axvline(threshold, color='black', linestyle='--', label=f'Threshold={threshold:.4f}')
    ax2.set_xlabel('Reconstruction Error (MSE)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Test Set')
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(prep_dir / "lstm_anomaly_reconstruction_error.png", dpi=150)
    print(f"\n✓ Reconstruction error plot saved")
    plt.close()


def main():
    """Main function"""
    try:
        print("\n" + "="*60)
        print("LSTM AUTOENCODER - ANOMALY DETECTION")
        print("="*60)
        
        # Load data
        X_train, X_test, X_val, y_train, y_test, y_val, label_mapping = load_preprocessed_data()
        
        if X_train is None:
            return
        
        # Verify training data is ONLY Normal
        if (y_train == 1).sum() > 0:
            print(f"\n⚠️  WARNING: Training set contains {(y_train == 1).sum()} anomaly samples!")
            print("   Autoencoder should train ONLY on Normal data.")
            return
        
        print(f"\n✓ Verified: Training set contains ONLY Normal data")
        
        # Scale features
        X_train_scaled, X_test_scaled, X_val_scaled, scaler = scale_features(X_train, X_test, X_val)
        
        # Build model
        input_shape = (WINDOW_SIZE, X_train_scaled.shape[2])
        model = build_lstm_autoencoder(input_shape)
        
        # Train autoencoder
        history = train_autoencoder(model, X_train_scaled, X_val_scaled, y_val)
        
        # Find optimal threshold
        threshold, mse_val = find_threshold(model, X_val_scaled, y_val)
        
        # Evaluate on test set
        accuracy, y_pred, mse_test = evaluate_anomaly_detection(model, X_test_scaled, y_test, threshold)
        
        # Plot reconstruction error distributions
        plot_reconstruction_error_distribution(mse_val, y_val, mse_test, y_test, threshold)
        
        # Save final model
        prep_dir = Path("prep")
        model.save(prep_dir / "lstm_anomaly_final.keras")
        print(f"\n✓ Final model saved")
        
        print("\n" + "="*60)
        print("✅ LSTM ANOMALY DETECTION COMPLETED!")
        print("="*60)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Threshold: {threshold:.6f}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

