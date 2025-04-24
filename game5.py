import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)

BLOCK_SIZE = 20
SPEED = 50

class SnakeGameAI:

    def __init__(self, w=640, h=480, square_count=3, rect_count=2, square_blocks=4, rect_dims=(8, 3)):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # Nombre d'obstacles et tailles en blocs
        self.square_count = square_count
        self.rect_count = rect_count
        self.square_blocks = square_blocks
        self.rect_dims = rect_dims  # (width_blocks, height_blocks)
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.frame_iteration = 0

        self._generate_obstacles()
        self.food = None
        self._place_food()

    def _generate_obstacles(self):
        self.obstacles = []
        # Zone autour du serpent initial à éviter
        center_zone = pygame.Rect(
            self.w//2 - self.square_blocks*BLOCK_SIZE,
            self.h//2 - self.square_blocks*BLOCK_SIZE,
            self.square_blocks*2*BLOCK_SIZE,
            self.square_blocks*2*BLOCK_SIZE
        )
        # Créer les carrés
        for _ in range(self.square_count):
            size = self.square_blocks * BLOCK_SIZE
            placed = False
            while not placed:
                bx = random.randint(0, (self.w - size) // BLOCK_SIZE) * BLOCK_SIZE
                by = random.randint(0, (self.h - size) // BLOCK_SIZE) * BLOCK_SIZE
                rect = pygame.Rect(bx, by, size, size)
                if rect.colliderect(center_zone):
                    continue
                if any(rect.colliderect(o) for o in self.obstacles):
                    continue
                self.obstacles.append(rect)
                placed = True
        # Créer les rectangles
        rw = self.rect_dims[0] * BLOCK_SIZE
        rh = self.rect_dims[1] * BLOCK_SIZE
        for _ in range(self.rect_count):
            placed = False
            while not placed:
                bx = random.randint(0, (self.w - rw) // BLOCK_SIZE) * BLOCK_SIZE
                by = random.randint(0, (self.h - rh) // BLOCK_SIZE) * BLOCK_SIZE
                rect = pygame.Rect(bx, by, rw, rh)
                if rect.colliderect(center_zone):
                    continue
                if any(rect.colliderect(o) for o in self.obstacles):
                    continue
                self.obstacles.append(rect)
                placed = True

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            food_rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
            if self.food not in self.snake and not any(food_rect.colliderect(obs) for obs in self.obstacles):
                break

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            reward = -0.1
            self.snake.pop()

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
        if any(pt_rect.colliderect(obs) for obs in self.obstacles):
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        for obs in self.obstacles:
            pygame.draw.rect(self.display, GREY, obs)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir
        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)