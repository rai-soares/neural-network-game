import pygame
import random

# Inicializa o pygame e configura a tela
pygame.init()
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Desvie dos Objetos!')

class Game:
  # Constantes do jogo
  WIDTH = 400
  HEIGHT = 600
  player_size = 50
  object_size = 40
  player_speed = 7
  WHITE = (255, 255, 255)
  BLACK = (0, 0, 0)
  RED = (255, 0, 0)
  # Cores para diferenciar os jogadores
  player_colors = [
    (0, 0, 0),      # preto
    (0, 0, 255),    # azul
    (0, 200, 0),    # verde
    (200, 0, 0),    # vermelho
    (200, 200, 0),  # amarelo
    (200, 0, 200),  # magenta
    (0, 200, 200),  # ciano
    (100, 100, 100) # cinza
  ]

  def __init__(self):
    # Inicializa tela e relógio do jogo
    pygame.init()
    self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
    pygame.display.set_caption('Desvie dos Objetos!')
    self.clock = pygame.time.Clock()

  def handle_events(self):
    # Processa eventos do pygame (fecha janela se necessário)
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return False
    return True

  def create_object(self):
    # Cria um bloco (linha) com um espaço (gap) aleatório para o player passar
    gap_width = self.player_size * 1.2
    gap_x = random.randint(0, self.WIDTH - int(gap_width))
    y = -self.object_size
    segments = []
    # Segmento à esquerda do gap
    if gap_x > 0:
      segments.append([0, y, gap_x, self.object_size])
    # Segmento à direita do gap
    if gap_x + gap_width < self.WIDTH:
      segments.append([gap_x + gap_width, y, self.WIDTH - (gap_x + gap_width), self.object_size])
    return segments

  def spawn_object(self, objects):
    # Adiciona um novo bloco se não houver ou se o anterior já desceu
    if not objects or all(seg[1] > self.object_size * 2 for seg in objects):
      segments = self.create_object()
      for seg in segments:
        objects.append(seg)
    return objects

  def move_objects(self, objects, object_speed):
    # Move os blocos para baixo
    for seg in objects:
      seg[1] += object_speed * 1.3
    return objects

  def check_objects(self, objects, score):
    # Remove blocos que saíram da tela e atualiza score
    new_objects = []
    for obj in objects:
      if obj[1] >= self.HEIGHT:
        score += 1
      else:
        new_objects.append(obj)
    return new_objects, score

  def check_collision(self, objects, player_x, player_y):
    # Verifica se o player colidiu com algum bloco
    for seg in objects:
      x, y, w, h = seg
      if (player_x < x + w and
        player_x + self.player_size > x and
        player_y < y + h and
        player_y + self.player_size > y):
        return True
    return False

  def draw_game(self, players, objects):
    # Desenha jogadores, blocos e scores na tela
    self.screen.fill(self.WHITE)
    for idx, player in enumerate(players):
      if player['alive']:
        color = self.player_colors[idx % len(self.player_colors)]
        pygame.draw.rect(self.screen, color, (player['x'], player['y'], self.player_size, self.player_size))
    for seg in objects:
      x, y, w, h = seg
      pygame.draw.rect(self.screen, self.RED, (x, y, w, h))
    font = pygame.font.SysFont(None, 36)
    scores_text = ' | '.join([f'P{idx+1}:{p["score"]}' for idx, p in enumerate(players)])
    score_text = font.render(scores_text, True, self.BLACK)
    self.screen.blit(score_text, (10, 10))
    pygame.display.flip()

  def run(self, num_players=5):
    # Executa uma simulação do jogo com jogadores aleatórios
    object_speed = 5
    objects = []
    frame_count = 0
    players = []
    for _ in range(num_players):
      players.append({
        'x': self.WIDTH // 2 - self.player_size // 2,
        'y': self.HEIGHT - self.player_size - 10,
        'score': 0,
        'alive': True
      })
    running = True
    while running:
      frame_count += 1
      if frame_count % 100 == 0:
        object_speed += 1
      running = self.handle_events()
      for player in players:
        if not player['alive']:
          continue
        move = random.choice([-1, 0, 1])
        if move == -1 and player['x'] > 0:
          player['x'] -= self.player_speed
        elif move == 1 and player['x'] < self.WIDTH - self.player_size:
          player['x'] += self.player_speed
        collision = self.check_collision(objects, player['x'], player['y'])
        if collision:
          player['alive'] = False
        else:
          player['score'] += 1
      objects = self.spawn_object(objects)
      objects = self.move_objects(objects, object_speed)
      running = any(p['alive'] for p in players)
      self.draw_game(players, objects)
      self.clock.tick(30)
    print('Scores:', [p['score'] for p in players])
