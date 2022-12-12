import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Définition des entrées du réseau

x = np.array([[1, 2], [3, 4], [5, 6]])

# Création du modèle de réseau de neurones en utilisant la bibliothèque Keras

model = Sequential()
model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilation et entraînement du modèle en utilisant l'algorithme de descente de gradient

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=10, batch_size=1)

# Évaluation du modèle sur des données de test

scores = model.evaluate(x, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
