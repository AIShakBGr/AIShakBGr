import pygame
import random
import numpy as np
import os

class SnakeEnv:
    def __init__(self, width=480, height=360, block_size=20, headless=False, max_steps=300):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.init()
        else:
            pygame.init()

        self.width = width
        self.height = height
        self.block = block_size
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()

        t, w, h = block_size, width, height
        self.obstacles = [
            pygame.Rect(t, t, t, t*3),
            pygame.Rect(t, t, t*3, t),
            pygame.Rect(w-2*t, t, t, t*3),
            pygame.Rect(w-4*t, t, t*3, t),
            pygame.Rect(t, h-4*t, t, t*3),
            pygame.Rect(t, h-2*t, t*3, t),
            pygame.Rect(w-2*t, h-4*t, t, t*3),
            pygame.Rect(w-4*t, h-2*t, t*3, t),
            pygame.Rect(w//4, h//3, w//2, t),
            pygame.Rect(w//4, 2*h//3, w//2, t)
        ]

        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [80, 50], [60, 50]]
        self.direction = 'RIGHT'
        self.score = 0
        self.steps_since_fruit = 0
        self.fruit_spawn = False
        self._place_fruit()
        self.fruit_spawn = True
        return self._get_observation()

    def step(self, action):
        if action == 0 and self.direction != 'DOWN': self.direction = 'UP'
        elif action == 1 and self.direction != 'UP': self.direction = 'DOWN'
        elif action == 2 and self.direction != 'RIGHT': self.direction = 'LEFT'
        elif action == 3 and self.direction != 'LEFT': self.direction = 'RIGHT'

        if self.direction == 'UP': self.snake_pos[1] -= self.block
        if self.direction == 'DOWN': self.snake_pos[1] += self.block
        if self.direction == 'LEFT': self.snake_pos[0] -= self.block
        if self.direction == 'RIGHT': self.snake_pos[0] += self.block

        self.snake_pos[0] %= self.width
        self.snake_pos[1] %= self.height
        self.snake_body.insert(0, list(self.snake_pos))

        reward = 0
        done = False
        self.steps_since_fruit += 1

        head_rect = pygame.Rect(*self.snake_pos, self.block, self.block)
        if any(head_rect.colliderect(obs) for obs in self.obstacles) or self.snake_body[0] in self.snake_body[1:]:
            return self._get_observation(), -10, True, {'score': self.score}

        if self.snake_pos == self.fruit_pos:
            reward = 10
            self.score += 1
            self.fruit_spawn = False
            self.steps_since_fruit = 0
        else:
            self.snake_body.pop()

        if not self.fruit_spawn:
            self._place_fruit()
            self.fruit_spawn = True

        if self.steps_since_fruit > self.max_steps:
            return self._get_observation(), -5, True, {'score': self.score, 'timeout': True}

        return self._get_observation(), reward, done, {'score': self.score}

    def render(self):
        self.window.fill((0, 0, 0))
        for obs in self.obstacles:
            pygame.draw.rect(self.window, (128, 128, 128), obs)
        for seg in self.snake_body:
            pygame.draw.rect(self.window, (0, 255, 0), pygame.Rect(seg[0], seg[1], self.block, self.block))
        pygame.draw.rect(self.window, (255, 255, 255), pygame.Rect(self.fruit_pos[0], self.fruit_pos[1], self.block, self.block))
        pygame.display.flip()
        self.clock.tick(15)

    def _place_fruit(self):
        while True:
            pos = [
                random.randrange(1, self.width // self.block) * self.block,
                random.randrange(1, self.height // self.block) * self.block
            ]
            rect = pygame.Rect(pos[0], pos[1], self.block, self.block)
            if not any(rect.colliderect(o) for o in self.obstacles) and pos not in self.snake_body:
                self.fruit_pos = pos
                break

    def _get_observation(self):
        # Exemple simple de vecteur d’état : direction + position relative du fruit
        head_x, head_y = self.snake_pos
        fruit_dx = (self.fruit_pos[0] - head_x) / self.width
        fruit_dy = (self.fruit_pos[1] - head_y) / self.height

        direction = [
            int(self.direction == 'UP'),
            int(self.direction == 'DOWN'),
            int(self.direction == 'LEFT'),
            int(self.direction == 'RIGHT')
        ]
        return np.array(direction + [fruit_dx, fruit_dy], dtype=np.float32)
