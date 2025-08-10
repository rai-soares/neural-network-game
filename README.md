# Neural Network Game

This project implements a reinforcement learning (RL) environment using Pygame, where multiple agents (players) learn to dodge falling obstacles using a custom neural network. The project includes real-time neural network visualization and a dynamic graph showing the network's structure and weights.

## Features

- **Multi-agent RL game**: Agents (players) try to survive by dodging falling blocks. The environment is rendered using Pygame.
- **Custom neural network**: A simple feedforward neural network (with configurable layers) is used for decision-making and trained via RL.
- **Real-time NN activation visualization**: The activations of the input, hidden, and output layers are shown live during training.
- **Graphical NN structure visualization**: The network's structure and weights are displayed as a graph, with edge colors and thickness representing weight values and sign.
- **Batch training**: Experiences from all epochs are collected and used for training at the end, improving learning stability.
- **Persistent training**: The best weights are saved and loaded automatically, allowing training to continue across runs.
- **Configurable parameters**: Number of players, epochs, network architecture, learning rate, and more can be easily adjusted in `index.py`.

## How it works

1. **Game loop**: Each agent is controlled by the neural network, which receives the current state and outputs an action (move left, stay, move right).
2. **Experience collection**: States, actions, and rewards are stored for each agent during each epoch.
3. **Training**: After all epochs, the neural network is trained using the collected experiences from all agents and epochs.
4. **Visualization**:
   - **NNVisualizer**: Shows activations of each layer in real time.
   - **NNGraphVisualizer**: Shows the network structure and updates edge colors/thickness as weights change.
5. **Saving/loading weights**: The best-performing weights are saved to disk and loaded at startup.

## Files

- `index.py`: Main RL loop, game setup, training logic, and integration with visualizers.
- `game.py`: Pygame-based game logic, rendering, and agent/environment interaction.
- `neural_network.py`: Custom neural network implementation (feedforward, backpropagation, training).
- `nn_visualization.py`: Contains NNVisualizer (activations) and NNGraphVisualizer (structure/weights).

## Usage

1. Install dependencies:
   ```bash
   pip install pygame matplotlib numpy
   ```
2. Run the main script:
   ```bash
   python3 index.py
   ```
3. Adjust parameters in `index.py` as needed (number of players, epochs, etc).

## Customization

- To change the neural network architecture, modify the parameters in `NeuralNetwork` in `index.py`.
- To visualize more layers, extend the neural network and pass all weight matrices to `NNGraphVisualizer`.
- You can enable/disable visualizations with the `SHOW_NN_VIS` flag in `index.py`.

## Project Structure

```
├── index.py
├── game.py
├── neural_network.py
├── nn_visualization.py
├── best_nn_weights.pkl
└── README.md
```

## Credits

- Pygame for game rendering
- Matplotlib for neural network visualization

## License

MIT License
