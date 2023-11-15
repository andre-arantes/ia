import numpy as np
# Aluno: André Arantes
# NeuralNetwork class: implementação de uma rede neural com uma camada oculta
# Referencia: https://gist.github.com/maunashjani/922e5a2a60130367dc58f1ee1fd6da36

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.1, n_epochs=1000, activation_function='sigmoid'):
        self.weights_input_hidden = np.random.rand(n_inputs, n_hidden)
        self.bias_hidden = np.random.rand(1, n_hidden)
        self.weights_hidden_output = np.random.rand(n_hidden, n_outputs)
        self.bias_output = np.random.rand(1, n_outputs)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation_function = activation_function

    def _activate(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)

    def _activate_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return x * (1 - x)
        elif self.activation_function == 'tanh':
            return 1 - x**2
        elif self.activation_function == 'relu':
            return (x > 0).astype(int)

    def train(self, inputs, labels):
        for _ in range(self.n_epochs):
            for i in range(len(inputs)):
                input_layer = inputs[i:i+1]
                hidden_layer_input = np.dot(
                    input_layer, self.weights_input_hidden) + self.bias_hidden
                hidden_layer_output = self._activate(hidden_layer_input)

                output_layer_input = np.dot(
                    hidden_layer_output, self.weights_hidden_output) + self.bias_output
                output_layer_output = self._activate(output_layer_input)

                output_error = labels[i:i+1] - output_layer_output
                output_delta = output_error * \
                    self._activate_derivative(output_layer_output)

                hidden_error = output_delta.dot(self.weights_hidden_output.T)
                hidden_delta = hidden_error * \
                    self._activate_derivative(hidden_layer_output)

                self.weights_hidden_output += self.learning_rate * \
                    hidden_layer_output.T.dot(output_delta)
                self.bias_output += self.learning_rate * \
                    np.sum(output_delta, axis=0, keepdims=True)
                self.weights_input_hidden += self.learning_rate * \
                    input_layer.T.dot(hidden_delta)
                self.bias_hidden += self.learning_rate * \
                    np.sum(hidden_delta, axis=0, keepdims=True)

    def predict(self, inputs):
        hidden_layer_input = np.dot(
            inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self._activate(hidden_layer_input)
        output_layer_input = np.dot(
            hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output_layer_output = self._activate(output_layer_input)
        return output_layer_output.round()

# Same test neural network used in the previous exercise (perceptron)
def test_neural_network(n_inputs, case, learning_rate, use_bias, activation_function):
    if case == "AND":
        inputs = np.array(
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
        )
        labels = np.array(
            [[0], [0], [0], [1]]
        )
    elif case == "OR":
        inputs = np.array(
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
        )
        labels = np.array(
            [[0], [1], [1], [1]]
        )
    elif case == "XOR":
        inputs = np.array(
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
        )
        labels = np.array(
            [[0], [1], [1], [0]]
        )

    neural_network = NeuralNetwork(
        n_inputs, 4, 1, learning_rate=learning_rate, activation_function=activation_function)
    if not use_bias:
        neural_network.bias_hidden = 0
        neural_network.bias_output = 0

    neural_network.train(inputs, labels)

    print(f"Teste da porta {case} com {n_inputs} entradas, taxa de Aprendizado: {learning_rate}, bias: {use_bias}, função de ativação: {activation_function}")
    for i in range(len(inputs)):
        result = neural_network.predict(inputs[i:i+1])
        print(f"Entrada: {inputs[i]} => Resultado: {result[0]}")


# Testing different configurations
# Using sigmoid activation function
test_neural_network(2, "AND", learning_rate=0.1,
                    use_bias=True, activation_function='sigmoid')
test_neural_network(2, "OR", learning_rate=0.1,
                    use_bias=True, activation_function='sigmoid')
test_neural_network(2, "XOR", learning_rate=0.1,
                    use_bias=True, activation_function='sigmoid')
# Using Tanh activation function
test_neural_network(2, "AND", learning_rate=0.1,
                    use_bias=True, activation_function='tanh')
test_neural_network(2, "OR", learning_rate=0.1,
                    use_bias=True, activation_function='tanh')
test_neural_network(2, "XOR", learning_rate=0.1,
                    use_bias=True, activation_function='tanh')
# Using ReLU activation function
test_neural_network(2, "AND", learning_rate=0.1,
                    use_bias=True, activation_function='relu')
test_neural_network(2, "OR", learning_rate=0.1,
                    use_bias=True, activation_function='relu')
test_neural_network(2, "XOR", learning_rate=0.1,
                    use_bias=True, activation_function='relu')
