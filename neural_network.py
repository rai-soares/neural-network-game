
import random
import math


class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size, lr=0.01, epochs=10):
    """
    Inicializa a rede neural com pesos aleatórios.
    input_size: número de entradas
    hidden_size: neurônios na camada oculta
    output_size: número de saídas (ações)
    lr: taxa de aprendizado
    epochs: épocas de treinamento por chamada
    """
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.lr = lr
    self.epochs = epochs
    self.W1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
    self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
    self.W2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
    self.b2 = [random.uniform(-1, 1) for _ in range(output_size)]

  def sigmoid(self, x):
    """Função de ativação sigmoid."""
    x = max(-60, min(60, x))
    return 1 / (1 + math.exp(-x))

  def sigmoid_deriv(self, x):
    """Derivada da função sigmoid."""
    sx = self.sigmoid(x)
    return sx * (1 - sx)

  def forward(self, x):
    """
    Executa o passo de feedforward:
    - Calcula ativação da camada oculta
    - Calcula ativação da camada de saída (softmax)
    Retorna valores intermediários para o backward.
    """
    z1 = [sum(x[i] * self.W1[j][i] for i in range(self.input_size)) + self.b1[j] for j in range(self.hidden_size)]
    a1 = [self.sigmoid(z) for z in z1]
    z2 = [sum(a1[i] * self.W2[j][i] for i in range(self.hidden_size)) + self.b2[j] for j in range(self.output_size)]
    # Softmax com estabilidade numérica
    max_z2 = max(z2)
    exp_z2 = [math.exp(v - max_z2) for v in z2]
    sum_exp = sum(exp_z2)
    if sum_exp == 0:
        a2 = [1.0 / self.output_size for _ in range(self.output_size)]
    else:
        a2 = [v / sum_exp for v in exp_z2]
    return z1, a1, z2, a2

  def mse(self, y_true, y_pred):
    """Calcula o erro quadrático médio entre saída e alvo."""
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

  def backward(self, x, y, z1, a1, z2, a2):
    # Executa o passo de backpropagation: calcula gradientes e atualiza pesos/bias
    dz2 = [a2[i] - y[i] for i in range(self.output_size)]
    dW2 = [[dz2[j] * a1[i] for i in range(self.hidden_size)] for j in range(self.output_size)]
    db2 = dz2[:]
    dz1 = [sum(dz2[j] * self.W2[j][i] for j in range(self.output_size)) * self.sigmoid_deriv(z1[i]) for i in range(self.hidden_size)]
    dW1 = [[dz1[j] * x[i] for i in range(self.input_size)] for j in range(self.hidden_size)]
    db1 = dz1[:]
    for j in range(self.output_size):
      for i in range(self.hidden_size):
        self.W2[j][i] -= self.lr * dW2[j][i]
    for j in range(self.output_size):
      self.b2[j] -= self.lr * db2[j]
    for j in range(self.hidden_size):
      for i in range(self.input_size):
        self.W1[j][i] -= self.lr * dW1[j][i]
    for j in range(self.hidden_size):
      self.b1[j] -= self.lr * db1[j]

  def train(self, X, Y):
    """
    Treina a rede neural nos dados X (entradas) e Y (saídas desejadas).
    Executa várias épocas de ajuste dos pesos.
    """
    for epoch in range(self.epochs):
      total_loss = 0
      for x, y in zip(X, Y):
        z1, a1, z2, a2 = self.forward(x)
        total_loss += self.mse(y, a2)
        self.backward(x, y, z1, a1, z2, a2)

  def predict(self, x):
    """
    Realiza uma predição para o estado x.
    Retorna o índice da ação com maior probabilidade.
    """
    _, _, _, a2 = self.forward(x)
    return a2.index(max(a2))
