import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load dataset
X_train, Y_train = load_data()

# Initialize model
model = Sequential()

# Add input layer
model.add(Dense(units=64, input_dim=64, activation='relu'))

# Add hidden layer
model.add(Dense(units=32, activation='relu'))

# Add output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, epochs=10, batch_size=32)
