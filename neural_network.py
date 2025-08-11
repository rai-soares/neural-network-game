
import random
import math

class NeuralNetwork:
  def train(self, X, Y):
    """
    Trains the neural network on data X (inputs) and Y (desired outputs).
    Performs multiple epochs of weight adjustment using backpropagation for arbitrary hidden layers.
    """
    for epoch in range(self.epochs):
      total_loss = 0
      for x, y in zip(X, Y):
        zs, activations = self.forward(x)
        # Calculate output error (MSE derivative)
        delta = [activations[-1][i] - y[i] for i in range(self.output_size)]
        deltas = [delta]
        # Backpropagate through hidden layers
        for l in range(len(self.hidden_sizes), 0, -1):
          layer = activations[l]
          z = zs[l-1]
          next_delta = []
          for i in range(len(layer)):
            error = 0.0
            for j in range(len(deltas[0])):
              error += deltas[0][j] * self.weights[l][j][i]
            next_delta.append(error * self.sigmoid_deriv(z[i]))
          deltas.insert(0, next_delta)
        # Update weights and biases
        for l in range(len(self.weights)):
          for i in range(len(self.weights[l])):
            for j in range(len(self.weights[l][i])):
              self.weights[l][i][j] -= self.lr * deltas[l][i] * activations[l][j]
            self.biases[l][i] -= self.lr * deltas[l][i]
        total_loss += self.mse(y, activations[-1])
      # Optionally print loss per epoch
      # print(f"Epoch {epoch+1}, Loss: {total_loss/len(X)}")

  def predict(self, x):
    """
    Makes a prediction for state x.
    Returns the index of the action with the highest probability.
    """
    _, activations = self.forward(x)
    return activations[-1].index(max(activations[-1]))
  
  def __init__(self, input_size, hidden_sizes, output_size, lr=0.01, epochs=10):
    """
    Initializes the neural network with random weights.
    input_size: number of inputs
    hidden_sizes: list of neurons in each hidden layer
    output_size: number of outputs (actions)
    lr: learning rate
    epochs: training epochs per call
    """
    self.input_size = input_size
    self.hidden_sizes = hidden_sizes
    self.output_size = output_size
    self.lr = lr
    self.epochs = epochs
    self.weights = []
    self.biases = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(len(layer_sizes)-1):
      self.weights.append([[random.uniform(-1, 1) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])])
      self.biases.append([random.uniform(-1, 1) for _ in range(layer_sizes[i+1])])

  def sigmoid(self, x):
    """Sigmoid activation function."""
    x = max(-60, min(60, x))
    return 1 / (1 + math.exp(-x))

  def sigmoid_deriv(self, x):
    """Derivative of the sigmoid function."""
    sx = self.sigmoid(x)
    return sx * (1 - sx)

  def forward(self, x):
    """
    Performs the feedforward step for arbitrary hidden layers.
    Returns all z and a activations for visualization and backward.
    """
    activations = [x]
    zs = []
    for w, b in zip(self.weights[:-1], self.biases[:-1]):
      z = [sum(activations[-1][i] * w[j][i] for i in range(len(activations[-1]))) + b[j] for j in range(len(w))]
      zs.append(z)
      a = [self.sigmoid(val) for val in z]
      activations.append(a)
    # Output layer (softmax)
    w_out = self.weights[-1]
    b_out = self.biases[-1]
    z_out = [sum(activations[-1][i] * w_out[j][i] for i in range(len(activations[-1]))) + b_out[j] for j in range(len(w_out))]
    max_z = max(z_out)
    exp_z = [math.exp(v - max_z) for v in z_out]
    sum_exp = sum(exp_z)
    if sum_exp == 0:
      a_out = [1.0 / len(z_out) for _ in range(len(z_out))]
    else:
      a_out = [v / sum_exp for v in exp_z]
    zs.append(z_out)
    activations.append(a_out)
    return zs, activations

  def mse(self, y_true, y_pred):
    """Calculates the mean squared error between output and target."""
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

  # NOTE: The backward, train, and predict methods below only support 1-2 hidden layers.
  # To support arbitrary hidden layers, these methods must be refactored to loop through all layers.
  # For now, only the forward pass supports arbitrary layers.
