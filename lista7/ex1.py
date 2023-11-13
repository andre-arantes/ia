import numpy as np

# Classe do Perceptron padrão
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1, n_epochs=1000):
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def train(self, inputs, labels):
        for _ in range(self.n_epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = labels[i] - prediction
                self.weights += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum > 0 else 0

# Função de teste do perceptron


def test_perceptron(n_inputs, case):
    if case == "AND":
        inputs = np.array(
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
        )
        labels = np.array(
            [0, 0, 0, 1]
        )
    elif case == "OR":
        inputs = np.array(
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
        )
        labels = np.array(
            [0, 1, 1, 1]
        )
    elif case == "XOR":
        inputs = np.array(
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
        )
        labels = np.array(
            [0, 1, 1, 0]
        )

    perceptron = Perceptron(n_inputs)
    perceptron.train(inputs, labels)

    print(f"Teste da porta {case} com {n_inputs} entradas:")
    for i in range(len(inputs)):
        result = perceptron.predict(inputs[i])
        print(f"Entrada: {inputs[i]} => Resultado: {result}")


test_perceptron(2, "AND")
test_perceptron(2, "OR")
test_perceptron(2, "XOR")
