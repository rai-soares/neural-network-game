import os
import pickle
import numpy as np
from nn_visualization import NNVisualizer, NNGraphVisualizer
from game import Game
from neural_network import NeuralNetwork
from nn_visualization import NNVisualizer

# USE 60 WITH NN_VISUALIZATION ON
GAME_TICK_RATE = 1000

def save_best_weights(nn, filename):
    """Salva os pesos da rede neural em um arquivo."""
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'wb') as f:
        pickle.dump({
            'weights': nn.weights,
            'biases': nn.biases
        }, f)
        print(f"[INFO] Pesos salvos em {filename}")

def load_best_weights(nn, filename):
    """Carrega os pesos salvos para a rede neural, se existirem."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            nn.weights = data['weights']
            nn.biases = data['biases']

def run_epoch(game, nn, num_players, object_speed, nn_vis=None, epsilon=0.1):
    """Executa uma época do jogo e coleta experiências dos jogadores."""
    objects = []
    frame_count = 0
    players = []
    # Initialize players at different positions, evenly spaced
    for i in range(num_players):
        if num_players > 1:
            x_pos = int((game.WIDTH - game.player_size) * i / (num_players - 1))
        else:
            x_pos = game.WIDTH // 2
        players.append({
            'x': x_pos,
            'y': game.HEIGHT - game.player_size - 10,
            'score': 0,
            'alive': True
        })
    memories = [ {'states': [], 'actions': [], 'rewards': []} for _ in range(num_players) ]
    running = True
    while running:
        frame_count += 1
        if frame_count % 100 == 0:
            object_speed += 1
        running = game.handle_events()
        for idx, player in enumerate(players):
            if not player['alive']:
                continue
            # Player state
            if objects:
                closest = min(objects, key=lambda o: abs(o[1] - player['y']))
                x_diff = (closest[0] - player['x']) / game.WIDTH
                y_diff = (closest[1] - player['y']) / game.HEIGHT
                segs_same_y = [seg for seg in objects if seg[1] == closest[1]]
                segs_sorted = sorted(segs_same_y, key=lambda s: s[0])
                if len(segs_sorted) == 2:
                    gap_x = segs_sorted[0][0] + segs_sorted[0][2]
                    gap_width = segs_sorted[1][0] - gap_x
                    gap_x_norm = gap_x / game.WIDTH
                    gap_width_norm = gap_width / game.WIDTH
                    gap_center_x = gap_x + gap_width / 2
                    gap_center_x_norm = gap_center_x / game.WIDTH
                    dist_to_gap_center = (gap_center_x - player['x']) / game.WIDTH
                else:
                    gap_x_norm = 0
                    gap_width_norm = 0
                    gap_center_x_norm = 0
                    dist_to_gap_center = 0
                state = [
                    player['x'] / game.WIDTH,
                    player['y'] / game.HEIGHT,
                    x_diff,
                    y_diff,
                    object_speed / game.HEIGHT,
                    gap_x_norm,
                    gap_width_norm,
                    gap_center_x_norm,
                    dist_to_gap_center
                ]
            else:
                state = [player['x'] / game.WIDTH, player['y'] / game.HEIGHT, 0, 0, object_speed / game.HEIGHT, 0, 0, 0, 0]
            # Forward NN and action
            # Check if the state size is correct
            if hasattr(nn, 'input_size') and len(state) != nn.input_size:
                print(f"[ERROR] Tamanho do estado ({len(state)}) diferente do esperado pela NN ({nn.input_size}). Estado: {state}")
                raise ValueError(f"Tamanho do estado ({len(state)}) diferente do esperado pela NN ({nn.input_size})")
            zs, activations = nn.forward(state)
            if nn_vis and idx == 0 and frame_count % 10 == 0:
                nn_vis.update(*activations)
            a_out = activations[-1]
            # Action selection
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1, 2])
            else:
                action = a_out.index(max(a_out))
            if 'epsilon' in locals():
                eps = epsilon
            else:
                eps = 0.1
            if np.random.rand() < eps:
                action = np.random.choice([0, 1, 2])
            else:
                action = a_out.index(max(a_out))
            memories[idx]['states'].append(state)
            memories[idx]['actions'].append(action)
            # Player movement
            if action == 0 and player['x'] > 0:
                player['x'] -= game.player_speed
            elif action == 2 and player['x'] < game.WIDTH - game.player_size:
                player['x'] += game.player_speed
            # Check collision
            collision = game.check_collision(objects, player['x'], player['y'])
            if collision:
                player['alive'] = False
                reward = -200  # Negative reward for collision
            else:
                # Reward only if passing through the gap, penalty if missed, neutral for survival
                if objects:
                    segs_same_y = [seg for seg in objects if seg[1] == closest[1]]
                    segs_sorted = sorted(segs_same_y, key=lambda s: s[0])
                    if len(segs_sorted) == 2:
                        gap_x = segs_sorted[0][0] + segs_sorted[0][2]
                        gap_width = segs_sorted[1][0] - gap_x
                        gap_center_x = gap_x + gap_width / 2
                        player_center_x = player['x'] + game.player_size / 2
                        dist_to_center = abs(player_center_x - gap_center_x)
                        dist_norm = dist_to_center / (gap_width / 2)
                        # Recompensa sempre proporcional à proximidade do centro do gap
                        reward = 100 * (1 - dist_norm)
                        if player['x'] + game.player_size > gap_x and player['x'] < gap_x + gap_width:
                            player['score'] += 1
                        else:
                            reward -= 50  # penalidade extra se não passar pelo gap
                    else:
                        player['score'] += 1
                        reward = 0  # Neutral for surviving
                else:
                    player['score'] += 0
                    reward = 0  # Small positive for surviving each frame
            memories[idx]['rewards'].append(reward)        
        # Update objects and screen
        objects = game.spawn_object(objects)
        objects = game.move_objects(objects, object_speed)
        # Keeps the window open until the user closes it manually
        running = any(p['alive'] for p in players)  # Removed to prevent automatic window closing200
        game.draw_game(players, objects)
        game.clock.tick(GAME_TICK_RATE)
    return players, memories

def train_nn(nn, memories, num_players, epoch):
    """Prepares the data and trains the neural network."""
    all_states = []
    all_Y = []
    for idx in range(num_players):
        states = memories[idx]['states']
        actions = memories[idx]['actions']
        rewards = memories[idx]['rewards']
        for a, r, s in zip(actions, rewards, states):
            y = [0, 0, 0]
            y[a] = r
            all_states.append(s)
            all_Y.append(y)
    print(f"Treinando NN na época {epoch} com {len(all_states)} amostras...", flush=True)
    nn.epochs = 100
    nn.train(all_states, all_Y)
    nn.epochs = 10
    print(f"Treino NN finalizado na época {epoch}.", flush=True)

def main():
    NUM_PLAYERS =  1
    NUM_EPOCHS = 2000  # Mais épocas para melhor aprendizado
    SHOW_NN_VIS = False  # do not use True if you have many players
    scores = []
    weights_file = 'best_nn_weights.pkl'
    # Use 2 hidden layers for more stable learning
    hidden_layers = [32, 32, 16]
    nn = NeuralNetwork(input_size=9, hidden_sizes=hidden_layers, output_size=3, lr=0.01, epochs=200)
    # Check if the weights file exists and if the architecture has changed
    if os.path.exists(weights_file):
        try:
            with open(weights_file, 'rb') as f:
                data = pickle.load(f)
                # Check if the shape of the weights is compatible
                if len(data['weights'][0][0]) != nn.input_size:
                    os.remove(weights_file)
                else:
                    load_best_weights(nn, weights_file)
        except Exception as e:
            print(f"[INFO] Erro ao carregar pesos: {e}. Removendo arquivo.")
            os.remove(weights_file)
    else:
        load_best_weights(nn, weights_file)
    nn_vis = NNVisualizer(input_size=9, hidden_sizes=hidden_layers, output_size=3) if SHOW_NN_VIS else None
    nn_graph_vis = NNGraphVisualizer(nn.weights) if SHOW_NN_VIS else None
    all_memories = []
    best_score_overall = -float('inf')
    best_weights = None
    epsilon_start = 0.8  # Mais exploração no início
    epsilon_end = 0.05
    nn.epochs = 200  # Mais épocas de treino por batch
    game = Game()
    for epoch in range(NUM_EPOCHS):
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (epoch / (NUM_EPOCHS * 0.8))
        if hasattr(game, 'reset'):
            game.reset()
        object_speed = 5
        players, memories = run_epoch(game, nn, NUM_PLAYERS, object_speed, nn_vis=nn_vis, epsilon=epsilon)
        all_memories.append(memories)
        player_scores = [p['score'] for p in players]
        best_score = max(player_scores)
        scores.append(best_score)
        print(f'Epoch {epoch+1} - Best score: {best_score}')
        if best_score > best_score_overall:
            best_score_overall = best_score
            best_weights = {
                'weights': nn.weights,
                'biases': nn.biases
            }
        if nn_graph_vis:
            nn_graph_vis.update(nn.weights)
    # Save weights to file (always saves the best of the session)
    save_best_weights(nn, weights_file)

if __name__ == "__main__":
    main()
