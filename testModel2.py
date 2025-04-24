import torch
import pygame
import numpy as np
import random
from model import Linear_QNet
from game6 import SnakeGameAI, Direction, Point

# Charger le modèle entraîné
model = Linear_QNet(11, 256, 3)
model.load_state_dict(torch.load('model/model_random2.pth'))
model.eval()

def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger droit
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        # Danger à droite
        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # Danger à gauche
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),

        # Direction actuelle
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # Position de la nourriture
        game.food.x < head.x,  # food à gauche
        game.food.x > head.x,  # food à droite
        game.food.y < head.y,  # food en haut
        game.food.y > head.y   # food en bas
    ]
    return np.array(state, dtype=int)

def test(num_games=20, random_prob=0.05):
    for i in range(num_games):
        game = SnakeGameAI()
        print(f"\n🎮 Partie {i+1} --------------------------")
        while True:
            state = get_state(game)
            state0 = torch.tensor(state, dtype=torch.float)

            # Prédiction du modèle
            prediction = model(state0)
            move = torch.argmax(prediction).item()

            # 🎲 Avec une probabilité, mouvement aléatoire
            if random.random() < random_prob:
                move = random.randint(0, 2)

            final_move = [0, 0, 0]
            final_move[move] = 1

            reward, done, score = game.play_step(final_move)

            if done:
                print(f"Game over. Score final: {score}")
                pygame.time.wait(1000)
                break

if __name__ == '__main__':
    test()
