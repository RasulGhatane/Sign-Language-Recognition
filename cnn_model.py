import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os

def count_subdirectories(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

# Data directories
train_dir = 'mydata/training_set'
test_dir = 'mydata/test_set'

# Count classes
num_classes = count_subdirectories(train_dir)
print(f"Number of classes detected: {num_classes}")

def create_improved_model(input_shape=(64, 64, 3), num_classes=29):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Custom data generator
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, target_size, class_mode='categorical', shuffle=True):
        self.generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        ).flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=shuffle
        )
        self.n = len(self.generator)
        self.batch_size = batch_size

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        X, y = self.generator[index]
        if y.shape[1] == 1:
            y = to_categorical(y, num_classes=num_classes)
        return X, y

# Create data generators
train_generator = CustomDataGenerator(train_dir, batch_size=64, target_size=(64, 64), class_mode='categorical')
test_generator = CustomDataGenerator(test_dir, batch_size=64, target_size=(64, 64), class_mode='categorical')

print("Class indices:", train_generator.generator.class_indices)

# Verify data shape
X_sample, y_sample = train_generator[0]
print(f"Sample batch image shape: {X_sample.shape}")
print(f"Sample batch label shape: {y_sample.shape}")
print(f"Sample labels: {y_sample[0]}")

# Create and compile the improved model
model = create_improved_model(num_classes=num_classes)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Training the model
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
    )

    # Saving the model
    model.save('Improved_Trained_model.h5')

    # Plotting training and validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plotting training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    # Additional debugging information
    print("Model output shape:", model.outputs[0].shape)
    print("Last layer output shape:", model.layers[-1].output.shape)
    
    # Check a batch of data
    batch_x, batch_y = next(iter(train_generator))
    print("Batch X shape:", batch_x.shape)
    print("Batch Y shape:", batch_y.shape)
    print("Sample Y values:", batch_y[0])  

# Print model configuration
print("Model configuration:")
print(model.get_config())