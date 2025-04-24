import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from model import Linear_QNet

class Agent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=1000):
        self.n_games = 0
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.memory = [] # Simple list for now, consider using a replay buffer
        self.model = Linear_QNet(state_size, 64, action_size) # hidden_size = 64
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)
        self.state_size = state_size
        self.action_size = action_size

    def get_state(self, game):
        head_x, head_y = game.snake_position
        fruit_x, fruit_y = game.fruit_position
        direction = game.direction

        # Calculer la direction relative du fruit
        fruit_direction_x = np.sign(fruit_x - head_x)
        fruit_direction_y = np.sign(fruit_y - head_y)

        # Vérifier les dangers immédiats (collision imminente)
        danger_front = game._is_collision(game._get_next_head_position(direction))
        danger_left = game._is_collision(game._get_next_head_position(game._turn_left(direction)))
        danger_right = game._is_collision(game._get_next_head_position(game._turn_right(direction)))

        # Encodage de la direction
        direction_encoding = [0, 0, 0, 0]  # UP, DOWN, LEFT, RIGHT
        if direction == 'UP': direction_encoding[0] = 1
        elif direction == 'DOWN': direction_encoding[1] = 1
        elif direction == 'LEFT': direction_encoding[2] = 1
        elif direction == 'RIGHT': direction_encoding[3] = 1

        state = np.array([
            fruit_direction_x,
            fruit_direction_y,
            int(danger_front),
            int(danger_left),
            int(danger_right),
            *direction_encoding
        ], dtype=int)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000: # Train on a larger batch occasionally
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random move vs. exploit
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.n_games / self.epsilon_decay)
        final_move = [0, 0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def load(self, filename='model.pth'):
        self.model.load(filename)

    def save(self, filename='model.pth'):
        self.model.save(filename)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    # Exemple d'utilisation de l'agent (très basique)
    state_size = 6 + 4
    action_size = 4
    agent = Agent(state_size, action_size)
    print("Agent initialized")