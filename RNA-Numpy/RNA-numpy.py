import numpy as np

# Définition des paramètres du réseau de neurones
n_inputs = 784  # Nombre d'entrées (une image de 28x28 pixels)
n_hidden1 = 256  # Nombre de neurones dans la première couche cachée
n_hidden2 = 128  # Nombre de neurones dans la seconde couche cachée
n_outputs = 10  # Nombre de sorties (un chiffre de 0 à 9)

# Initialisation des poids et des biais
weights = {
    'hidden1': np.random.randn(n_inputs, n_hidden1),
    'hidden2': np.random.randn(n_hidden1, n_hidden2),
    'output': np.random.randn(n_hidden2, n_outputs)
}

biases = {
    'hidden1': np.zeros(n_hidden1),
    'hidden2': np.zeros(n_hidden2),
    'output': np.zeros(n_outputs)
}

# Fonction d'activation : sigmoïde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Fonction de propagation avant (forward propagation)
def forward_prop(x, weights, biases):
    hidden1_out = sigmoid(np.dot(x, weights['hidden1']) + biases['hidden1'])
    hidden2_out = sigmoid(np.dot(hidden1_out, weights['hidden2']) + biases['hidden2'])
    output_out = sigmoid(np.dot(hidden2_out, weights['output']) + biases['output'])
    return output_out

# Exemple d'utilisation du réseau
input_data = np.random.randn(1, 784)  # Génère des données d'entrée aléatoires
output = forward_prop(input_data, weights, biases)  # Propagation avant
print(output)  # Affiche la sortie du réseau
