import pygame
import random
from enum import Enum
from collections import namedtuple
from collections import deque
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Couleurs
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)

BLOCK_SIZE = 20
SPEED = 500

class SnakeGameAI:

    def __init__(self, w=640, h=480, square_count=3, rect_count=2, square_blocks=4, rect_dims=(8, 3)):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.square_count = square_count
        self.rect_count = rect_count
        self.square_blocks = square_blocks
        self.rect_dims = rect_dims
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2*BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.frame_iteration = 0
        self._generate_obstacles()
        self.food = None
        self._place_food()
        self.last_positions = deque(maxlen=15)

    def _generate_obstacles(self):
        self.obstacles = []
        center_zone = pygame.Rect(
            self.w//2 - self.square_blocks*BLOCK_SIZE,
            self.h//2 - self.square_blocks*BLOCK_SIZE,
            self.square_blocks*2*BLOCK_SIZE,
            self.square_blocks*2*BLOCK_SIZE
        )
        
        # Génération des carrés
        for _ in range(self.square_count):
            size = self.square_blocks * BLOCK_SIZE
            while True:
                bx = random.randint(0, (self.w - size)//BLOCK_SIZE) * BLOCK_SIZE
                by = random.randint(0, (self.h - size)//BLOCK_SIZE) * BLOCK_SIZE
                rect = pygame.Rect(bx, by, size, size)
                if not rect.colliderect(center_zone) and not any(rect.colliderect(o) for o in self.obstacles):
                    self.obstacles.append(rect)
                    break
        
        # Génération des rectangles
        rw = self.rect_dims[0] * BLOCK_SIZE
        rh = self.rect_dims[1] * BLOCK_SIZE
        for _ in range(self.rect_count):
            while True:
                bx = random.randint(0, (self.w - rw)//BLOCK_SIZE) * BLOCK_SIZE
                by = random.randint(0, (self.h - rh)//BLOCK_SIZE) * BLOCK_SIZE
                rect = pygame.Rect(bx, by, rw, rh)
                if not rect.colliderect(center_zone) and not any(rect.colliderect(o) for o in self.obstacles):
                    self.obstacles.append(rect)
                    break

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            food_rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
            if self.food not in self.snake and not any(food_rect.colliderect(obs) for obs in self.obstacles):
                break

    def play_step(self, action):
        self.frame_iteration += 1
        self.last_positions.append(self.head)
        reward = -0.1
        game_over = False
        
        # Détection de boucle
        if len(self.last_positions) > 10:
            unique = len(set(self.last_positions))
            if unique < 5:
                reward -= 3
                self.frame_iteration += 5

        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Mouvement
        self._move(action)
        self.snake.insert(0, self.head)

        # Vérification collision
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = -12
            game_over = True
            return reward, game_over, self.score

        # Manger la nourriture
        if self.head == self.food:
            self.score += 1
            reward = 15
            self._place_food()
        else:
            self.snake.pop()

        # Mise à jour interface
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        pt_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
        return any(pt_rect.colliderect(obs) for obs in self.obstacles)

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        for obs in self.obstacles:
            pygame.draw.rect(self.display, GREY, obs)
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        # Nouveau système de priorité de mouvement
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # continuer droit
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # droite
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4  # gauche
            new_dir = clock_wise[next_idx]

        self.direction = new_dir
        x, y = self.head
        if new_dir == Direction.RIGHT: x += BLOCK_SIZE
        elif new_dir == Direction.LEFT: x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN: y += BLOCK_SIZE
        elif new_dir == Direction.UP: y -= BLOCK_SIZE
        
        self.head = Point(x, y)
