import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Check GPU availability
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and will be used.")
else:
    print("GPU is not available. Ensure CUDA and cuDNN are properly installed.")

# Define paths
base_path = '/Training_Set/Training'
info_csv_path = '/Training_Set/RFMiD_Training_Labels.csv'

# Define paths for evaluation dataset
eval_base_path = '/RetinalDatase/Evaluation_Set/Evaluation_Set/Validation'
eval_info_csv_path = 'RetinalDataset/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv'

# Load Kaggle Data
if not os.path.exists(info_csv_path):
    raise FileNotFoundError(f"The file {info_csv_path} does not exist. Please check the path.")

kaggle_info = pd.read_csv(info_csv_path)

# Load Evaluation Data
if not os.path.exists(eval_info_csv_path):
    raise FileNotFoundError(f"The file {eval_info_csv_path} does not exist. Please check the path.")

eval_info = pd.read_csv(eval_info_csv_path)

# Preprocess images
def load_and_preprocess_image(img_id, dataset_base_path):
    img_path = os.path.join(dataset_base_path, f"{img_id}.png")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        img = cv2.resize(img, (128, 128))  # Smaller size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img / 255.0
    print(f"Image not found: {img_path}")
    return None

def load_data(info_df, dataset_base_path):
    images = []
    labels = []
    for _, row in info_df.iterrows():
        img_id = row['ID']
        img = load_and_preprocess_image(img_id, dataset_base_path)
        if img is not None:
            images.append(img)
            label = np.array([
                row['Disease_Risk'], row['DR'], row['ARMD'], row['MH'], row['DN'],
                row['MYA'], row['BRVO'], row['TSLN'], row['ERM']
            ])
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess training and evaluation data
images, labels = load_data(kaggle_info, base_path)
eval_images, eval_labels = load_data(eval_info, eval_base_path)

if images.size == 0:
    raise ValueError("No images were loaded for training. Please check the image paths and preprocessing steps.")
if eval_images.size == 0:
    raise ValueError("No images were loaded for evaluation. Please check the image paths and preprocessing steps.")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(images)

# Build a simplified CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Smaller input size and fewer filters
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(labels.shape[1], activation='sigmoid')
])

# Compile the model
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Learning rate scheduler with warm-up
def scheduler(epoch, lr):
    if epoch < 10:
        return lr * (epoch + 1) / 10
    else:
        return lr * 0.9

lr_scheduler = LearningRateScheduler(scheduler)

# Callbacks
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Fit the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),  # Reduced batch size
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=[early_stopping, reduce_lr, lr_scheduler]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Evaluate the model on the evaluation dataset
eval_loss, eval_acc = model.evaluate(eval_images, eval_labels)
print(f'Evaluation accuracy: {eval_acc}')

# Save the model
model_save_path = '/CNNFromScratch.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()