import os
import pickle
from game import Game
from neural_network import NeuralNetwork
from nn_visualization import NNVisualizer

def save_best_weights(nn, filename):
    """Salva os pesos da rede neural em um arquivo."""
    with open(filename, 'wb') as f:
        pickle.dump({
            'W1': nn.W1,
            'b1': nn.b1,
            'W2': nn.W2,
            'b2': nn.b2
        }, f)

def load_best_weights(nn, filename):
    """Carrega os pesos salvos para a rede neural, se existirem."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            nn.W1 = data['W1']
            nn.b1 = data['b1']
            nn.W2 = data['W2']
            nn.b2 = data['b2']

def run_epoch(game, nn, num_players, object_speed, nn_vis=None):
    """Executa uma época do jogo e coleta experiências dos jogadores."""
    objects = []
    frame_count = 0
    players = []
    # Inicializa os jogadores em posições diferentes, espaçados uniformemente
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
            # Estado do jogador
            if objects:
                closest = min(objects, key=lambda o: o[1])
                x_diff = (closest[0] - player['x']) / game.WIDTH
                y_diff = (closest[1] - player['y']) / game.HEIGHT
                state = [player['x'] / game.WIDTH, player['y'] / game.HEIGHT, x_diff, y_diff, object_speed / 20]
            else:
                state = [player['x'] / game.WIDTH, player['y'] / game.HEIGHT, 0, 0, object_speed / 20]
            # Forward NN e ação
            z1, a1, z2, a2 = nn.forward(state)
            if nn_vis and idx == 0 and frame_count % 500 == 0:
                nn_vis.update(state, a1, a2)
            action = a2.index(max(a2))
            memories[idx]['states'].append(state)
            memories[idx]['actions'].append(action)
            # Movimento do jogador
            if action == 0 and player['x'] > 0:
                player['x'] -= game.player_speed
            elif action == 2 and player['x'] < game.WIDTH - game.player_size:
                player['x'] += game.player_speed
            # Verifica colisão
            collision = game.check_collision(objects, player['x'], player['y'])
            if collision:
                player['alive'] = False
                memories[idx]['rewards'].append(-30)
            else:
                # Recompensa se passar pelo espaço livre
                if objects:
                    segs = [seg for seg in objects if seg[1] < game.HEIGHT // 2]
                    if segs:
                        segs_sorted = sorted(segs, key=lambda s: s[0])
                        if len(segs_sorted) == 2:
                            gap_x = segs_sorted[0][0] + segs_sorted[0][2]
                            gap_width = segs_sorted[1][0] - gap_x
                            if player['x'] + game.player_size > gap_x and player['x'] < gap_x + gap_width:
                                player['score'] += 1
                                memories[idx]['rewards'].append(20)
                            else:
                                memories[idx]['rewards'].append(0)
                        else:
                            player['score'] += 1
                            memories[idx]['rewards'].append(1)
                    else:
                        player['score'] += 1
                        memories[idx]['rewards'].append(1)
                else:
                    player['score'] += 1
                    memories[idx]['rewards'].append(1)
        # Atualiza objetos e tela
        objects = game.spawn_object(objects)
        objects = game.move_objects(objects, object_speed)
        # Mantém a janela aberta até o usuário fechar manualmente
        running = any(p['alive'] for p in players)  # Removido para não fechar automaticamente
        game.draw_game(players, objects)
        game.clock.tick(500)
    return players, memories

def train_nn(nn, memories, num_players, epoch):
    """Prepara os dados e treina a rede neural."""
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
    nn.train(all_states, all_Y)
    print(f"Treino NN finalizado na época {epoch}.", flush=True)

def main():
    NUM_PLAYERS = 1
    NUM_EPOCHS = 50
    SHOW_NN_VIS = True  # do not use True if you have many players
    scores = []
    weights_file = 'best_nn_weights.pkl'
    nn = NeuralNetwork(input_size=5, hidden_size=10, output_size=3, lr=0.05, epochs=5)
    load_best_weights(nn, weights_file)
    nn_vis = NNVisualizer(input_size=5, hidden_size=10, output_size=3) if SHOW_NN_VIS else None
    best_score_overall = -float('inf')
    best_weights = None
    for epoch in range(NUM_EPOCHS):
        game = Game()
        object_speed = 5
        players, memories = run_epoch(game, nn, NUM_PLAYERS, object_speed, nn_vis=nn_vis)
        player_scores = [p['score'] for p in players]
        best_score = max(player_scores)
        scores.append(best_score)
        print(f'Época {epoch+1} - Melhor score: {best_score}')
        # Save the best weights
        if best_score > best_score_overall:
            best_score_overall = best_score
            best_weights = {
                'W1': nn.W1,
                'b1': nn.b1,
                'W2': nn.W2,
                'b2': nn.b2
            }
        # Train the neural network
        train_nn(nn, memories, NUM_PLAYERS, epoch)
        # Save weights to file
        if best_weights:
            save_best_weights(nn, weights_file)

if __name__ == "__main__":
    main()
