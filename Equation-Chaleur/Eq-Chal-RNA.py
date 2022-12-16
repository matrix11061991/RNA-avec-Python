import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Constants
dx = 0.1
dt = 0.01
alpha = 0.1

# Initialize the temperature grid
T = np.zeros((100, 100))
T[:, 0] = 100  # Left edge is 100 degrees
T[:, -1] = 0  # Right edge is 0 degrees

# Build the neural network model
model = Sequential()
model.add(Dense(100, input_shape=(100,), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.compile(loss='mse', optimizer='adam')

# Iterate over time steps
for t in range(100):
  # Compute the new temperature grid
  T_new = np.zeros_like(T)
  for i in range(1, 99):
    for j in range(1, 99):
      T_new[i, j] = T[i, j] + alpha * dt / dx**2 * (T[i+1, j] - 2*T[i, j] + T[i-1, j] + T[i, j+1] - 2*T[i, j] + T[i, j-1])
  
  # Use the neural network to predict the temperature at the next time step
  predictions = model.predict(T_new)
  
  # Update the temperature grid with the predicted values
  T = predictions
