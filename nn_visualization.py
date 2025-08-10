import matplotlib.pyplot as plt
import numpy as np

class NNVisualizer:
    """
    Visualizes neural network activations in real time.
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fig, self.axs = plt.subplots(1, 3, figsize=(10, 3))
        self.fig.suptitle('Ativações da Rede Neural')
        self.input_bar = self.axs[0].bar(range(input_size), np.zeros(input_size))
        self.axs[0].set_title('Input')
        self.hidden_bar = self.axs[1].bar(range(hidden_size), np.zeros(hidden_size))
        self.axs[1].set_title('Hidden')
        self.output_bar = self.axs[2].bar(range(output_size), np.zeros(output_size))
        self.axs[2].set_title('Output')
        plt.ion()
        plt.show()

    def update(self, input_vec, hidden_vec, output_vec):
        for bar, val in zip(self.input_bar, input_vec):
            bar.set_height(val)
        for bar, val in zip(self.hidden_bar, hidden_vec):
            bar.set_height(val)
        for bar, val in zip(self.output_bar, output_vec):
            bar.set_height(val)
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
