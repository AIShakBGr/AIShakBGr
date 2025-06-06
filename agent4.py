import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 250_000
BATCH_SIZE = 2000
LR = 0.0005

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 80  # Exploration initiale élevée
        self.gamma = 0.85  # Taux d'actualisation réduit
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(19, 512, 3)  # 11 + 8 = 19 entrées
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Détection obstacles dans 8 directions
        obstacle = []
        for dx, dy in [(-20,0), (20,0), (0,-20), (0,20),
                      (-20,-20), (20,-20), (-20,20), (20,20)]:
            pt = Point(head.x + dx, head.y + dy)
            obstacle.append(game.is_collision(pt))

        state = [
            # Danger immédiat
            (dir_r and game.is_collision(Point(head.x+20, head.y))) or
            (dir_l and game.is_collision(Point(head.x-20, head.y))) or
            (dir_u and game.is_collision(Point(head.x, head.y-20))) or
            (dir_d and game.is_collision(Point(head.x, head.y+20))),

            # Danger à droite
            (dir_u and game.is_collision(Point(head.x+20, head.y))) or
            (dir_d and game.is_collision(Point(head.x-20, head.y))) or
            (dir_l and game.is_collision(Point(head.x, head.y-20))) or
            (dir_r and game.is_collision(Point(head.x, head.y+20))),

            # Danger à gauche
            (dir_d and game.is_collision(Point(head.x+20, head.y))) or
            (dir_u and game.is_collision(Point(head.x-20, head.y))) or
            (dir_r and game.is_collision(Point(head.x, head.y-20))) or
            (dir_l and game.is_collision(Point(head.x, head.y+20))),

            # Direction
            dir_l, dir_r, dir_u, dir_d,

            # Nourriture
            game.food.x < head.x,  # gauche
            game.food.x > head.x,  # droite
            game.food.y < head.y,  # haut
            game.food.y > head.y,  # bas

            # Obstacles (8 directions)
            *obstacle
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(25, 120 - self.n_games)  # Exploration minimale de 25%
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(square_count=2, rect_count=1)  # Moins d'obstacles initiaux

    while True:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} | Score {score} | Record {record} | Epsilon {agent.epsilon}')

            plot_scores.append(score)
            total_score += score
            mean_score = np.mean(plot_scores[-20:]) if len(plot_scores) >= 20 else np.mean(plot_scores)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
