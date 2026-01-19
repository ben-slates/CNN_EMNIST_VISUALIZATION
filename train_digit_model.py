import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

print(" Loading MNIST dataset...")
# Load MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f" Dataset shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"  Classes: {np.unique(y_train)} (0-9)")

# Create validation set from training data
val_split = 0.1  # 10% for validation
val_size = int(len(X_train) * val_split)
X_val = X_train[:val_size]
y_val = y_train[:val_size]
X_train_final = X_train[val_size:]
y_train_final = y_train[val_size:]

print(f"\n Train/Validation split:")
print(f"  Training samples: {len(X_train_final)}")
print(f"  Validation samples: {len(X_val)}")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    shear_range=0.08
)

print("\n Building CNN model with regularization...")
# Enhanced CNN with regularization
model = models.Sequential([
    # First conv block
    layers.Conv2D(32, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005),
                  padding='same',
                  input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005),
                  padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.2),

    # Second conv block
    layers.Conv2D(64, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005),
                  padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005),
                  padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.3),

    # Third conv block (added for deeper learning)
    layers.Conv2D(128, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005),
                  padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005),
                  padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.4),

    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(10, activation='softmax')
])

# Learning rate schedule
initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=12,  # Wait 12 epochs without improvement
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'digit_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("\n Training model for 15 epochs (with early stopping)...")
print("   Using data augmentation and regularization to prevent overfitting")

# Train with data augmentation
history = model.fit(
    datagen.flow(X_train_final, y_train_final, batch_size=128),
    epochs=15,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print(f"\n Actual epochs trained: {len(history.history['loss'])}")

# Evaluate on test set
print("\n Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f" Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f" Test Loss: {test_loss:.4f}")

# Additional metrics
print("\n Additional metrics:")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Save the final model
model.save("digit_model.h5")
print("\n Model saved as 'digit_model.h5'")
print(" Best model during training saved as 'digit_model_best.h5'")

# Plot training history
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training & validation loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate if available
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('digit_training_history.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(" Training history plot saved as 'digit_training_history.png'")

plot_training_history(history)

# Test a few sample predictions
print("\n🔍 Sample predictions:")
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    image = X_test[idx]
    true_label = y_test[idx]
    prediction = model.predict(image.reshape(1, 28, 28, 1))
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    print(f"  Image {idx}: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.2%}")

print(f"\n Training complete!")
print(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"   Final test accuracy: {test_acc:.4f}")