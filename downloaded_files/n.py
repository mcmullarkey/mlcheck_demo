import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load data from the CSV file
data = pd.read_csv('synthetic_dataset.csv')

# Separate features and labels
X = data.drop(['Target','Subject_id'], axis=1).values
y = data['Target'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data to the appropriate format for RNN (num_samples, time_steps, num_features)
# In this example, we will use 50 time steps, assuming each sample has 50 time steps
time_steps = 50
num_features = X.shape[1]
X_train = X_train.reshape(-1, X_train.shape[0], num_features)
X_test = X_test.reshape(-1, X_test.shape[0], num_features)

# Create the RNN model
model = models.Sequential()
model.add(layers.SimpleRNN(64, input_shape=(time_steps, num_features), activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
epochs = 10
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
