import os

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pygame
import time

# =====================
# 1. Environment Wrapper
# =====================
class SnakeEnv:
    def __init__(self, width=480, height=360, block_size=10, headless=False):
        # Mode affichage
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

        # Obstacles (crochets et barres)
        t, w, h = block_size, width, height
        self.obstacles = [
            pygame.Rect(t, t, t, t*3),            # coin haut-gauche vertical
            pygame.Rect(t, t, t*3, t),            # coin haut-gauche horizontal
            pygame.Rect(w-2*t, t, t, t*3),        # coin haut-droit vertical
            pygame.Rect(w-4*t, t, t*3, t),        # coin haut-droit horizontal
            pygame.Rect(t, h-4*t, t, t*3),        # coin bas-gauche vertical
            pygame.Rect(t, h-2*t, t*3, t),        # coin bas-gauche horizontal
            pygame.Rect(w-2*t, h-4*t, t, t*3),    # coin bas-droit vertical
            pygame.Rect(w-4*t, h-2*t, t*3, t),    # coin bas-droit horizontal
            pygame.Rect(w//4, h//3, w//2, t),     # barre milieu haut
            pygame.Rect(w//4, 2*h//3, w//2, t)    # barre milieu bas
        ]
        self.reset()

    def reset(self):
        self.snake_pos = [100, 50]
        self.snake_body = [[100,50],[90,50],[80,50]]
        self.direction = 'RIGHT'
        self.score = 0
        # Génération de la position du fruit en évitant obstacles
        self.fruit_spawn = False
        self._place_fruit()
        self.fruit_spawn = True
        return self._get_observation()

    def step(self, action):
        # 0=UP,1=DOWN,2=LEFT,3=RIGHT
        if action == 0 and self.direction != 'DOWN':  self.direction = 'UP'
        elif action == 1 and self.direction != 'UP':   self.direction = 'DOWN'
        elif action == 2 and self.direction != 'RIGHT':self.direction = 'LEFT'
        elif action == 3 and self.direction != 'LEFT': self.direction = 'RIGHT'

        # Déplacement
        if self.direction == 'UP':    self.snake_pos[1] -= self.block
        if self.direction == 'DOWN':  self.snake_pos[1] += self.block
        if self.direction == 'LEFT':  self.snake_pos[0] -= self.block
        if self.direction == 'RIGHT': self.snake_pos[0] += self.block

        # Wrap-around
        self.snake_pos[0] %= self.width
        self.snake_pos[1] %= self.height

        self.snake_body.insert(0, list(self.snake_pos))
        reward, done = 0, False

        # Collision obstacles
        head_rect = pygame.Rect(*self.snake_pos, self.block, self.block)
        for obs in self.obstacles:
            if head_rect.colliderect(obs):
                reward, done = -10, True
                break

        # Collision corps
        if self.snake_body[0] in self.snake_body[1:]:
            reward, done = -10, True

        # Manger fruit
        if self.snake_pos == self.fruit_pos:
            self.score += 1
            reward = 10
            self.fruit_spawn = False
        else:
            self.snake_body.pop()

        # Spawn fruit après consommation, évitant obstacles
        if not self.fruit_spawn:
            self._place_fruit()
            self.fruit_spawn = True

        return self._get_observation(), reward, done, {'score': self.score}

    def render(self):
        self.window.fill((0,0,0))
        for obs in self.obstacles:
            pygame.draw.rect(self.window, (128,128,128), obs)
        for seg in self.snake_body:
            pygame.draw.rect(self.window, (0,255,0), pygame.Rect(seg[0], seg[1], self.block, self.block))
        pygame.draw.rect(self.window, (255,255,255), pygame.Rect(self.fruit_pos[0], self.fruit_pos[1], self.block, self.block))
        pygame.display.flip()
        self.clock.tick(15)

    def _get_observation(self):
        # TODO: remplacer par vrai vecteur d'état
        return np.zeros(10, dtype=np.float32)

    def _place_fruit(self):
        while True:
            pos = [
                random.randrange(1, self.width//self.block) * self.block,
                random.randrange(1, self.height//self.block) * self.block
            ]
            rect = pygame.Rect(pos[0], pos[1], self.block, self.block)
            # ne place que là où il n'y a pas d'obstacle
            if not any(rect.colliderect(o) for o in self.obstacles):
                self.fruit_pos = pos
                break

# =====================
# 2. Réseau DQN
# =====================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

# =====================
# 3. Agent DQN
# =====================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.policy_net.fc[-1].out_features)
        with torch.no_grad():
            q = self.policy_net(torch.from_numpy(state).unsqueeze(0))
        return q.argmax().item()

    def remember(self, s,a,r,s2,done):
        self.memory.append((s,a,r,s2,done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = np.array(states,      dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions     = np.array(actions,     dtype=np.int64)
        rewards     = np.array(rewards,     dtype=np.float32)
        dones       = np.array(dones,       dtype=np.float32)

        states      = torch.from_numpy(states)
        next_states = torch.from_numpy(next_states)
        actions     = torch.from_numpy(actions)
        rewards     = torch.from_numpy(rewards)
        dones       = torch.from_numpy(dones)

        q_vals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0]
        target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_vals, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =====================
# 4. Boucle d'entraînement
# =====================
if __name__ == '__main__':
    env = SnakeEnv(headless=False)
    state_dim  = 10  # à adapter
    action_dim = 4
    agent = DQNAgent(state_dim, action_dim)
    episodes = 500
    target_update = 10

    for ep in range(1, episodes+1):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            if steps % 4 == 0:
                agent.replay()

            # Affichage à chaque pas
            env.render()

            state = next_state
            total_reward += reward
            steps += 1

        if ep % target_update == 0:
            agent.update_target()

        print(f"Episode {ep} | Score {info['score']} | Reward total {total_reward} | Epsilon {agent.epsilon:.3f}")

    pygame.quit()