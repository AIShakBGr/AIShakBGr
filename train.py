import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 40
COLORS = {
    'background': (0, 0, 0),
    'snake_head': (0, 255, 0),
    'snake_body': (0, 200, 0),
    'food': (255, 0, 0),
    'obstacle': (150, 75, 0)  # Couleur marron pour les obstacles
}

class SnakeEnv:
    def __init__(self, w=640, h=480, render=False):
        self.w = w
        self.h = h
        self.render = render
        self.action_space = [0, 1, 2]  # [Tout droit, Droite, Gauche]
        self.observation_space = (12,)
        self.obstacles = []
        
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake RL')
        else:
            self.display = None
            
        self._place_obstacles()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self._get_state()

    def _place_obstacles(self):
        # Carré central de 5x5 blocs
        center_x = self.w // 2
        center_y = self.h // 2
        for i in range(-2, 3):
            for j in range(-2, 3):
                if abs(i) == 2 or abs(j) == 2:  # Contour uniquement
                    self.obstacles.append(Point(center_x + i*BLOCK_SIZE, 
                                              center_y + j*BLOCK_SIZE))

    def _get_state(self):
        head = self.head
        food = self.food
        
        state = [
            # Danger dans les 4 directions
            self._is_collision(Point(head.x + BLOCK_SIZE, head.y)),
            self._is_collision(Point(head.x - BLOCK_SIZE, head.y)),
            self._is_collision(Point(head.x, head.y - BLOCK_SIZE)),
            self._is_collision(Point(head.x, head.y + BLOCK_SIZE)),
            
            # Direction actuelle
            self.direction == Direction.RIGHT,
            self.direction == Direction.LEFT,
            self.direction == Direction.UP,
            self.direction == Direction.DOWN,
            
            # Position nourriture
            food.x < head.x,
            food.x > head.x,
            food.y < head.y,
            food.y > head.y
        ]
        
        return np.array(state, dtype=int)

    def step(self, action):
        self.frame_iteration += 1
        reward = 0
        game_over = False
        
        self._update_direction(action)
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return self._get_state(), reward, game_over, self.score
            
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.1
        
        if self.render:
            self._update_ui()
            pygame.time.wait(50)
        
        return self._get_state(), reward, game_over, self.score

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake and self.food not in self.obstacles:
                break

    def _is_collision(self, pt=None):
        pt = pt or self.head
        # Collision avec les bords
        if (pt.x >= self.w - BLOCK_SIZE or 
            pt.x < 0 or 
            pt.y >= self.h - BLOCK_SIZE or 
            pt.y < 0):
            return True
        # Collision avec le corps
        if pt in self.snake[1:]:
            return True
        # Collision avec les obstacles
        if pt in self.obstacles:
            return True
        return False

    def _update_ui(self):
        self.display.fill(COLORS['background'])
        
        # Dessiner les obstacles
        for pt in self.obstacles:
            pygame.draw.rect(self.display, COLORS['obstacle'], 
                           pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Dessiner le serpent
        for idx, pt in enumerate(self.snake):
            color = COLORS['snake_head'] if idx == 0 else COLORS['snake_body']
            pygame.draw.rect(self.display, color, 
                           pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Dessiner la nourriture
        pygame.draw.rect(self.display, COLORS['food'], 
                       pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT: x += BLOCK_SIZE
        elif direction == Direction.LEFT: x -= BLOCK_SIZE
        elif direction == Direction.DOWN: y += BLOCK_SIZE
        elif direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _update_direction(self, action):
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(self.direction)
        
        if action == 1: idx = (idx + 1) % 4  # Droite
        elif action == 2: idx = (idx - 1) % 4  # Gauche
        
        new_dir = directions[idx]
        if abs(directions.index(self.direction) - idx) != 2:
            self.direction = new_dir

class DQN(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size=12, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.model = DQN(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])
        
        targets = rewards + (1 - dones) * self.gamma * torch.max(self.model(next_states), dim=1)[0]
        predicted = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        loss = self.criterion(predicted, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train():
    env = SnakeEnv(render=True)  # Active le rendu
    agent = Agent(state_size=12, action_size=3)
    clock = pygame.time.Clock()
    
    for episode in range(500):
        state = env.reset()
        done = False
        score = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
            action = agent.act(state)
            next_state, reward, done, current_score = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score = current_score
            
            agent.replay()
            
            # Mise à jour visuelle
            env._update_ui()
            clock.tick(20)
            pygame.display.set_caption(f"Entraînement - Episode: {episode+1} | Score: {score} | Epsilon: {agent.epsilon:.2f}")
        
        print(f"Episode: {episode+1} | Score: {score}")
    
    torch.save(agent.model.state_dict(), 'snake_dqn_obstacles.pth')

if __name__ == '__main__':
    train()
