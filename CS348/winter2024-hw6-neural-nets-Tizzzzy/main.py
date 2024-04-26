import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 10
dropout_rate = 0
hidden_layer_size = 4096

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(hidden_layer_size, activation='relu'),
    Dropout(dropout_rate),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

import time
start_time = time.perf_counter()

# Train the model
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)

end_time = time.perf_counter()
time_lapse = end_time - start_time
print(f"Training Time: {time_lapse} seconds")

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest Accuracy:', test_acc)
