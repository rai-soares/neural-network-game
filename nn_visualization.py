import matplotlib.pyplot as plt
import numpy as np
class NNGraphVisualizer:
    def __init__(self, weights, biases=None):
        self.weights = weights  # list of weight matrices: [W1, W2, W3]
        self.biases = biases if biases is not None else []
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        plt.ion()
        plt.show()

    def update(self, weights, biases=None):
        self.weights = weights
        self.biases = biases if biases is not None else []
        self.ax.clear()
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        self.draw_network()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def draw_network(self):
        # Determine layer sizes from weights
        layer_sizes = [len(self.weights[0][0])] + [len(w) for w in self.weights]
        node_positions = []
        color_palette = ['#1976d2', '#26a69a', '#7e57c2', '#fbc02d', '#8d6e63', '#43a047']
        for i, size in enumerate(layer_sizes):
            x = i * 2.5
            y_positions = np.linspace(-2, 2, size)
            node_positions.append([(x, y) for y in y_positions])
            for (nx, ny) in node_positions[-1]:
                color = color_palette[i % len(color_palette)]
                self.ax.add_patch(plt.Circle((nx, ny), 0.18, color=color, ec='black', lw=1.5, zorder=2))
        # Draw weights between layers
        for l in range(len(self.weights)):
            W = self.weights[l]
            for i, (x0, y0) in enumerate(node_positions[l]):
                for j, (x1, y1) in enumerate(node_positions[l+1]):
                    w = W[j][i]
                    color = '#e53935' if w < 0 else '#1e88e5'
                    self.ax.plot([x0, x1], [y0, y1], color=color, linewidth=max(0.7, abs(w)*1.5), alpha=0.6, zorder=1)
        self.ax.set_xlim(-1, 2.5 * (len(layer_sizes)-1) + 1)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.axis('off')
        self.ax.set_title('Neural Network Structure', fontsize=16, fontweight='bold', pad=20)

class NNVisualizer:
    """
    Visualizes neural network activations in real time.
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        n_bars = 2 + len(hidden_sizes)  # input + hidden layers + output
        self.fig, self.axs = plt.subplots(1, n_bars, figsize=(2*n_bars, 3))
        self.fig.suptitle('Neural Network Activations')
        self.input_bar = self.axs[0].bar(range(input_size), np.zeros(input_size))
        self.axs[0].set_title('Input')
        self.hidden_bars = []
        for i, h_size in enumerate(hidden_sizes):
            bars = self.axs[i+1].bar(range(h_size), np.zeros(h_size))
            self.axs[i+1].set_title(f'Hidden {i+1}')
            self.hidden_bars.append(bars)
        self.output_bar = self.axs[-1].bar(range(output_size), np.zeros(output_size))
        self.axs[-1].set_title('Output')
        plt.ion()
        plt.show()

    def update(self, *activations):
        # activations: input, hidden1, hidden2, ..., output
        for bar, val in zip(self.input_bar, activations[0]):
            bar.set_height(val)
        for i, hidden_vec in enumerate(activations[1:-1]):
            for bar, val in zip(self.hidden_bars[i], hidden_vec):
                bar.set_height(val)
        for bar, val in zip(self.output_bar, activations[-1]):
            bar.set_height(val)
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
