import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

import keras
from keras import layers, Input, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
import torch
torch.manual_seed(42)

WINDOW_SIZE = 29

# -------------------------------
# 1. Load dataset
# -------------------------------
prep_dir = Path("prep")
X_train = np.load(prep_dir / "X_train_seq.npy")
y_train = np.load(prep_dir / "y_train_seq.npy")
X_test = np.load(prep_dir / "X_test_seq.npy")
y_test = np.load(prep_dir / "y_test_seq.npy")
X_val = np.load(prep_dir / "X_val_seq.npy")
y_val = np.load(prep_dir / "y_val_seq.npy")

with open(prep_dir / "label_mapping.pkl", "rb") as f:
    label_mapping = pickle.load(f)

# ----------------------------------------------------
# 2. FIXED: Correct sampling (10,000 per class TOTAL)
# ----------------------------------------------------
train_indices = []
for cls in range(5):
    cls_idx = np.where(y_train == cls)[0]
    sampled = np.random.choice(cls_idx, size=10000, replace=False)
    train_indices.extend(sampled)

train_indices = np.array(train_indices)
np.random.shuffle(train_indices)

X_train_sample = X_train[train_indices]
y_train_sample = y_train[train_indices]

# Test: 2000 per class
test_indices = []
for cls in range(5):
    cls_idx = np.where(y_test == cls)[0]
    sampled = np.random.choice(cls_idx, size=2000, replace=False)
    test_indices.extend(sampled)
test_indices = np.array(test_indices)
np.random.shuffle(test_indices)
X_test_sample = X_test[test_indices]
y_test_sample = y_test[test_indices]

# Validation: 1000 per class
val_indices = []
for cls in range(5):
    cls_idx = np.where(y_val == cls)[0]
    sampled = np.random.choice(cls_idx, size=1000, replace=False)
    val_indices.extend(sampled)
val_indices = np.array(val_indices)
np.random.shuffle(val_indices)
X_val_sample = X_val[val_indices]
y_val_sample = y_val[val_indices]

# -------------------------------
# 3. Standard scaling
# -------------------------------
scaler = StandardScaler()
n_train, window_size, n_features = X_train_sample.shape

X_train_scaled = scaler.fit_transform(
    X_train_sample.reshape(-1, n_features)
).reshape(n_train, window_size, n_features)

X_test_scaled = scaler.transform(
    X_test_sample.reshape(-1, n_features)
).reshape(X_test_sample.shape)

X_val_scaled = scaler.transform(
    X_val_sample.reshape(-1, n_features)
).reshape(X_val_sample.shape)

# -------------------------------
# 4. Class weights
# -------------------------------
classes = np.unique(y_train_sample)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_sample)
class_weight_dict = dict(zip(classes, class_weights))


# ----------------------------------------------------
# 5. Improved Transformer Model (fixed!)
# ----------------------------------------------------
def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=1)
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    # Attention + Norm
    x = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feed-forward block
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(inputs.shape[-1])(ff)

    return layers.LayerNormalization(epsilon=1e-6)(ff + x)


input_shape = (WINDOW_SIZE, X_train_scaled.shape[2])
inputs = Input(shape=input_shape)

# Add positional info
pos = layers.Embedding(input_dim=WINDOW_SIZE, output_dim=32)(tf.range(WINDOW_SIZE))
pos = tf.expand_dims(pos, 0)
pos = tf.tile(pos, [tf.shape(inputs)[0], 1, 1])
x = layers.Concatenate()([inputs, pos])

# Add multiple transformer blocks
for _ in range(2):
    x = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=64, dropout=0.3)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(5, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# 6. Training
# -------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)
]

history = model.fit(
    X_train_scaled, y_train_sample,
    validation_data=(X_val_scaled, y_val_sample),
    epochs=40,
    batch_size=256,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------
# 7. Plot Training History
# -------------------------------
print("\n7. Plotting training history...")
plot_dir = Path("prep/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
epochs = range(1, len(history.history['accuracy']) + 1)
ax1.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Plot loss
ax2.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = plot_dir / "training_history.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Combined plot saved: {plot_path}")

# Separate accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
accuracy_plot_path = plot_dir / "accuracy_plot.png"
plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Accuracy plot saved: {accuracy_plot_path}")

# Separate loss plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_plot_path = plot_dir / "loss_plot.png"
plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Loss plot saved: {loss_plot_path}")
print(f"\n   📊 All plots saved in: {plot_dir}/")

# -------------------------------
# 8. Evaluate
# -------------------------------
print("\n8. Evaluating on test set...")
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
print("Test accuracy:", accuracy_score(y_test_sample, y_pred))
print(classification_report(y_test_sample, y_pred))
