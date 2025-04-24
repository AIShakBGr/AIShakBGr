import pygame
import time
import random
import numpy as np  # Pour la manipulation de tableaux (états)

class SnakeGameRL:
    def __init__(self, window_x=480, window_y=360, block_size=10):
        pygame.init()
        self.window_x = window_x
        self.window_y = window_y
        self.block_size = block_size
        self.game_window = pygame.display.set_mode((self.window_x, self.window_y))
        pygame.display.set_caption('Snake Game RL')
        self.clock = pygame.time.Clock()
        self.snake_speed = 15  # Peut-être ajustable pour l'entraînement

        # Couleurs
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.grey = pygame.Color(128, 128, 128)

        self.reset()
        self.obstacle_thickness = self.block_size
        self.obstacles = [
            # Crochets dans les coins (adaptés à block_size)
            pygame.Rect(self.block_size, self.block_size, self.obstacle_thickness, self.obstacle_thickness * 3),
            pygame.Rect(self.block_size, self.block_size, self.obstacle_thickness * 3, self.obstacle_thickness),
            pygame.Rect(self.window_x - 2 * self.block_size, self.block_size, self.obstacle_thickness, self.obstacle_thickness * 3),
            pygame.Rect(self.window_x - 4 * self.block_size, self.block_size, self.obstacle_thickness * 3, self.obstacle_thickness),
            pygame.Rect(self.block_size, self.window_y - 4 * self.block_size, self.obstacle_thickness, self.obstacle_thickness * 3),
            pygame.Rect(self.block_size, self.window_y - 2 * self.block_size, self.obstacle_thickness * 3, self.obstacle_thickness),
            pygame.Rect(self.window_x - 2 * self.block_size, self.window_y - 4 * self.block_size, self.obstacle_thickness, self.obstacle_thickness * 3),
            pygame.Rect(self.window_x - 4 * self.block_size, self.window_y - 2 * self.block_size, self.obstacle_thickness * 3, self.obstacle_thickness),
            # Barres horizontales au milieu (adaptées à block_size)
            pygame.Rect(self.window_x // 4, self.window_y // 3, self.window_x // 2, self.obstacle_thickness),
            pygame.Rect(self.window_x // 4, 2 * self.window_y // 3, self.window_x // 2, self.obstacle_thickness)
        ]

    def reset(self):
        """Réinitialise l'état du jeu pour un nouvel épisode."""
        self.snake_position = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
        self.fruit_position = self._spawn_fruit()
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.game_over_flag = False
        return self._get_state()

    def _spawn_fruit(self):
        """Génère une position aléatoire pour le fruit qui n'est pas sur un obstacle."""
        while True:
            fruit_x = random.randrange(1, (self.window_x // self.block_size)) * self.block_size
            fruit_y = random.randrange(1, (self.window_y // self.block_size)) * self.block_size
            fruit_rect = pygame.Rect(fruit_x, fruit_y, self.block_size, self.block_size)
            is_on_obstacle = False
            for obstacle in self.obstacles:
                if fruit_rect.colliderect(obstacle):
                    is_on_obstacle = True
                    break
            if not is_on_obstacle:
                return [fruit_x, fruit_y]

    def _get_state(self):
        """Retourne l'état actuel du jeu sous forme de tableau numpy."""
        # Ici, tu définiras les caractéristiques de l'état que l'agent observera.
        # Exemples :
        # - Position relative du fruit par rapport à la tête du serpent (x, y)
        # - Présence d'un obstacle directement devant, à gauche et à droite (booléen)
        # - Direction actuelle du serpent (encodée)
        head_x, head_y = self.snake_position
        fruit_x, fruit_y = self.fruit_position
        direction = self.direction

        # Calculer la direction relative du fruit
        fruit_direction_x = np.sign(fruit_x - head_x)
        fruit_direction_y = np.sign(fruit_y - head_y)

        # Vérifier les dangers immédiats (collision imminente)
        danger_front = self._is_collision(self._get_next_head_position(direction))
        danger_left = self._is_collision(self._get_next_head_position(self._turn_left(direction)))
        danger_right = self._is_collision(self._get_next_head_position(self._turn_right(direction)))

        # Encodage de la direction
        direction_encoding = [0, 0, 0, 0]  # UP, DOWN, LEFT, RIGHT
        if direction == 'UP':
            direction_encoding[0] = 1
        elif direction == 'DOWN':
            direction_encoding[1] = 1
        elif direction == 'LEFT':
            direction_encoding[2] = 1
        elif direction == 'RIGHT':
            direction_encoding[3] = 1

        state = np.array([
            fruit_direction_x,
            fruit_direction_y,
            int(danger_front),
            int(danger_left),
            int(danger_right),
            *direction_encoding
        ], dtype=int)
        return state

    def _get_next_head_position(self, direction):
        """Calcule la prochaine position de la tête du serpent en fonction de la direction."""
        head_x, head_y = self.snake_position[0], self.snake_position[1]
        if direction == 'UP':
            return [head_x, head_y - self.block_size]
        elif direction == 'DOWN':
            return [head_x, head_y + self.block_size]
        elif direction == 'LEFT':
            return [head_x - self.block_size, head_y]
        elif direction == 'RIGHT':
            return [head_x + self.block_size, head_y]
        return [head_x, head_y]

    def _turn_left(self, direction):
        if direction == 'UP': return 'LEFT'
        if direction == 'DOWN': return 'RIGHT'
        if direction == 'LEFT': return 'DOWN'
        if direction == 'RIGHT': return 'UP'

    def _turn_right(self, direction):
        if direction == 'UP': return 'RIGHT'
        if direction == 'DOWN': return 'LEFT'
        if direction == 'LEFT': return 'UP'
        if direction == 'RIGHT': return 'DOWN'

    def step(self, action):
        """Effectue une action (changer de direction), met à jour l'état du jeu,
        et retourne la nouvelle observation, la récompense et si la partie est terminée."""
        self._take_action(action)
        self._move()

        reward = 0
        game_over = False

        if self._is_collision(self.snake_position):
            self.game_over_flag = True
            game_over = True
            reward = -10  # Pénalité pour la collision
        elif self.snake_position == self.fruit_position:
            self.score += 1
            reward = 10  # Récompense pour avoir mangé le fruit
            self.snake_body.insert(0, list(self.snake_position))
            self.fruit_position = self._spawn_fruit()
        else:
            self.snake_body.insert(0, list(self.snake_position))
            self.snake_body.pop()
            reward = 0.01 # Petite récompense pour survivre (encourager le mouvement)

        if len(self.snake_body) > 100: # Condition de fin alternative pour éviter des parties trop longues
            game_over = True
            reward = 0

        next_state = self._get_state()
        return next_state, reward, game_over, {} # Le dernier dict est pour info de debug

    def _take_action(self, action):
        """Traduit l'action de l'agent en un changement de direction."""
        if action == 0 and self.direction != 'DOWN':
            self.change_to = 'UP'
        elif action == 1 and self.direction != 'UP':
            self.change_to = 'DOWN'
        elif action == 2 and self.direction != 'RIGHT':
            self.change_to = 'LEFT'
        elif action == 3 and self.direction != 'LEFT':
            self.change_to = 'RIGHT'

        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

    def _move(self):
        """Met à jour la position de la tête du serpent."""
        if self.direction == 'UP':
            self.snake_position[1] -= self.block_size
        elif self.direction == 'DOWN':
            self.snake_position[1] += self.block_size
        elif self.direction == 'LEFT':
            self.snake_position[0] -= self.block_size
        elif self.direction == 'RIGHT':
            self.snake_position[0] += self.block_size

        # Gestion du passage à travers les bords (inchangé)
        if self.snake_position[0] < 0:
            self.snake_position[0] = self.window_x - self.block_size
        elif self.snake_position[0] >= self.window_x:
            self.snake_position[0] = 0
        if self.snake_position[1] < 0:
            self.snake_position[1] = self.window_y - self.block_size
        elif self.snake_position[1] >= self.window_y:
            self.snake_position[1] = 0

    def _is_collision(self, head_position):
        """Vérifie si la tête du serpent entre en collision avec les obstacles ou son corps."""
        head_rect = pygame.Rect(head_position[0], head_position[1], self.block_size, self.block_size)
        # Collision avec les obstacles
        for obstacle in self.obstacles:
            if head_rect.colliderect(obstacle):
                return True
        # Collision avec le corps
        for block in self.snake_body[1:]:
            if head_position[0] == block[0] and head_position[1] == block[1]:
                return True
        return False

    def render(self):
        """Affiche l'état actuel du jeu (pour visualisation pendant l'entraînement)."""
        self.game_window.fill(self.black)
        for obstacle in self.obstacles:
            pygame.draw.rect(self.game_window, self.grey, obstacle)
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
        pygame.draw.rect(self.game_window, self.white, pygame.Rect(self.fruit_position[0], self.fruit_position[1], self.block_size, self.block_size))
        score_font = pygame.font.SysFont('consolas', 20)
        score_surface = score_font.render('Score : ' + str(self.score), True, self.white)
        self.game_window.blit(score_surface, (10, 10))
        pygame.display.flip()

    def close(self):
        pygame.quit()

# Exemple d'utilisation de l'environnement (sans l'agent RL pour l'instant)
if __name__ == '__main__':
    env = SnakeGameRL()
    while not env.game_over_flag:
        state = env.reset()
        done = False
        while not done:
            # Agent prendrait une décision ici basée sur l'état
            action = random.randrange(0, 4) # Exemple d'action aléatoire
            next_state, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.1)
    env.close()