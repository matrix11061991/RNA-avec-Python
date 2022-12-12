import numpy as np

# Définition des poids et biais du réseau
weights = [
    np.array([[-1, 1], [1, 1]]),  # Couche 1 : 2 neurones, 2 entrées chacun
    np.array([[-1], [1]])         # Couche 2 : 1 neurone, 2 entrées
]
biases = [
    np.array([0, 0]), # Couche 1 : 2 neurones
    np.array([0])     # Couche 2 : 1 neurone
]

# Fonction d'activation sigmoïde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Propagation avant
def forward_propagation(inputs):
    # Couche 1 : calcul des activations en fonction des poids et des biais
    layer1_activations = sigmoid(np.dot(inputs, weights[0]) + biases[0])
    
    # Couche 2 : calcul des activations en fonction des poids et des biais
    layer2_activations = sigmoid(np.dot(layer1_activations, weights[1]) + biases[1])
    
    # Renvoi des résultats
    return layer1_activations, layer2_activations

# Exemple d'utilisation du réseau avec des données d'entrée
inputs = np.array([[0, 1]])
layer1_activations, layer2_activations = forward_propagation(inputs)
print(layer1_activations) # Attendu : [[0.5, 0.5]]
print(layer2_activations) # Attendu : [[0.5]]
