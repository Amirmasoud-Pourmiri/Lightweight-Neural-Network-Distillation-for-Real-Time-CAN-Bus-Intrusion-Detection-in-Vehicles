"""
ANN (Artificial Neural Network) Model for CAN Bus Multi-Class Anomaly Detection
"""
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
import torch
torch.manual_seed(42)

WINDOW_SIZE = 29


def load_preprocessed_data():
    """Load preprocessed sequential data"""
    print("="*60)
    print("LOADING PREPROCESSED SEQUENTIAL DATA")
    print("="*60)
    
    prep_dir = Path("prep")
    
    required_files = [
        "X_train_seq.npy", "y_train_seq.npy",
        "X_test_seq.npy", "y_test_seq.npy",
        "X_val_seq.npy", "y_val_seq.npy",
        "label_mapping.pkl"
    ]
    
    missing = [f for f in required_files if not (prep_dir / f).exists()]
    if missing:
        print(f"❌ Missing files: {missing}")
        print("Please run 'python prepare_canbus_multiclass.py' first.")
        return None, None, None, None, None, None, None
    
    print("Loading data...")
    X_train = np.load(prep_dir / "X_train_seq.npy")
    y_train = np.load(prep_dir / "y_train_seq.npy")
    X_test = np.load(prep_dir / "X_test_seq.npy")
    y_test = np.load(prep_dir / "y_test_seq.npy")
    X_val = np.load(prep_dir / "X_val_seq.npy")
    y_val = np.load(prep_dir / "y_val_seq.npy")
    
    with open(prep_dir / "label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)
    
    print(f"✓ Train: X={X_train.shape}, y={y_train.shape}")
    print(f"✓ Test:  X={X_test.shape}, y={y_test.shape}")
    print(f"✓ Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"✓ Labels: {label_mapping}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, label_mapping


def scale_features(X_train, X_test, X_val):
    """Scale features"""
    print("\nScaling features...")
    scaler = StandardScaler()
    
    n_train, window_size, n_features = X_train.shape
    n_test, n_val = X_test.shape[0], X_val.shape[0]
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, window_size, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_test, window_size, n_features)
    X_val_scaled = scaler.transform(X_val_flat).reshape(n_val, window_size, n_features)
    
    prep_dir = Path("prep")
    with open(prep_dir / "ann_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved to prep/ann_scaler.pkl")
    
    return X_train_scaled, X_test_scaled, X_val_scaled, scaler


def build_ann_model(input_shape, num_classes):
    """Build ANN model - flattens window and uses fully connected layers"""
    print("\n" + "="*60)
    print("BUILDING ANN MODEL")
    print("="*60)
    print(f"Input shape: {input_shape}")
    print(f"Output classes: {num_classes}")
    
    model = Sequential([
        # Flatten the window (29 x 12 = 348)
        Flatten(input_shape=input_shape),
        
        # First hidden layer
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Second hidden layer
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Third hidden layer
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Fourth hidden layer
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model"""
    print("\n" + "="*60)
    print("TRAINING ANN MODEL")
    print("="*60)
    
    prep_dir = Path("prep")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(prep_dir / "ann_best.keras"), monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        callbacks=callbacks,
        shuffle=False,  # Preserve temporal order
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, label_mapping):
    """Evaluate model"""
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    y_pred_proba = model.predict(X_test, batch_size=256, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Explicitly specify labels to handle cases where not all classes are predicted
    labels = sorted(label_mapping.keys())
    target_names = [label_mapping[i] for i in labels]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))
    
    prep_dir = Path("prep")
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(prep_dir / "ann_confusion_matrix.csv")
    print(f"\n✓ Confusion matrix saved")
    
    results_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'True_Attack': [label_mapping[i] for i in y_test],
        'Predicted_Attack': [label_mapping[i] for i in y_pred],
        'Correct': y_test == y_pred
    })
    results_df.to_csv(prep_dir / "ann_predictions.csv", index=False)
    print(f"✓ Predictions saved")
    
    print("\nPer-Class Accuracy:")
    for label, name in label_mapping.items():
        mask = y_test == label
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  {name}: {class_acc:.4f} ({mask.sum():,} samples)")
    
    return accuracy, y_pred


def main():
    """Main function"""
    try:
        print("\n" + "="*60)
        print("ANN - CAN BUS MULTI-CLASS ANOMALY DETECTION")
        print("="*60)
        
        X_train, X_test, X_val, y_train, y_test, y_val, label_mapping = load_preprocessed_data()
        
        if X_train is None:
            return
        
        X_train_scaled, X_test_scaled, X_val_scaled, scaler = scale_features(X_train, X_test, X_val)
        
        num_classes = len(label_mapping)
        input_shape = (WINDOW_SIZE, X_train_scaled.shape[2])
        
        model = build_ann_model(input_shape, num_classes)
        history = train_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
        accuracy, y_pred = evaluate_model(model, X_test_scaled, y_test, label_mapping)
        
        prep_dir = Path("prep")
        model.save(prep_dir / "ann_final.keras")
        print(f"\n✓ Final model saved to prep/ann_final.keras")
        
        print("\n" + "="*60)
        print("✅ ANN TRAINING COMPLETED!")
        print("="*60)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

