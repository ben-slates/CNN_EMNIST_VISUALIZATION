import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# IDX loaders
def load_images(path):
    with open(path,'rb') as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n,r,c)

def load_labels(path):
    with open(path,'rb') as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# Load EMNIST Letters
print(" Loading EMNIST Letters dataset...")
X_train = load_images("data/emnist_letters/emnist-letters-train-images-idx3-ubyte")
y_train = load_labels("data/emnist_letters/emnist-letters-train-labels-idx1-ubyte").copy()

X_test = load_images("data/emnist_letters/emnist-letters-test-images-idx3-ubyte")
y_test  = load_labels("data/emnist_letters/emnist-letters-test-labels-idx1-ubyte").copy()


# Fix rotation
print(" Fixing image orientation...")
X_train = np.array([np.fliplr(np.rot90(i,-1)) for i in X_train])
X_test = np.array([np.fliplr(np.rot90(i,-1)) for i in X_test])

# Labels 1–26 → 0–25
y_train -= 1
y_test -= 1

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print(f" Dataset shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"  Classes: {np.unique(y_train)} (0-25, A-Z)")

# Split training into train/validation sets
val_split = 0.2
val_size = int(len(X_train) * val_split)
X_val = X_train[:val_size]
y_val = y_train[:val_size]
X_train_final = X_train[val_size:]
y_train_final = y_train[val_size:]

print(f"\n Train/Validation split:")
print(f"  Training samples: {len(X_train_final)}")
print(f"  Validation samples: {len(X_val)}")

# Data Augmentation for regularization
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Improved CNN with regularization
print("\n Building CNN model with regularization...")
model = models.Sequential([
    # First conv block with BatchNorm and Dropout
    layers.Conv2D(32, (3,3), activation='relu', 
                  kernel_regularizer=regularizers.l2(0.001),
                  input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    # Second conv block
    layers.Conv2D(64, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    # Third conv block
    layers.Conv2D(128, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(26, activation='softmax')
])

# Optimizer with learning rate schedule
initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr, decay_steps=10000, decay_rate=0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,  # Wait 10 epochs without improvement
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by half
        patience=5,  # Wait 5 epochs
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'letter_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("\n Training model for 20 epochs (with early stopping)...")
print("   Using data augmentation and regularization to prevent overfitting")

# Train with data augmentation
history = model.fit(
    datagen.flow(X_train_final, y_train_final, batch_size=128),
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\n🧪 Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f" Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f" Test Loss: {test_loss:.4f}")

# Save the final model
model.save("letter_model.h5")
print(" Model saved as 'letter_model.h5'")
print(" Best model during training saved as 'letter_model_best.h5'")

# Plot training history (optional)
import matplotlib.pyplot as plt

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print(" Training history plot saved as 'training_history.png'")

plot_history(history)

print("\n Training complete!")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Total epochs actually trained: {len(history.history['loss'])}")