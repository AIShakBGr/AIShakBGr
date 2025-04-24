import torch
import pygame
import numpy as np
from model import Linear_QNet
from game3 import SnakeGameAI, Direction, Point

# Charger le modèle
model = Linear_QNet(11, 256, 3)
model.load_state_dict(torch.load('model/model1000.pth'))  # Assurez-vous que le chemin est correct
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
        # Danger straight
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),

        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # Food location
        game.food.x < head.x,  # food left
        game.food.x > head.x,  # food right
        game.food.y < head.y,  # food up
        game.food.y > head.y   # food down
    ]
    return np.array(state, dtype=int)

def test():
    game = SnakeGameAI()
    while True:
        state = get_state(game)
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = model(state0)
        move = torch.argmax(prediction).item()

        final_move = [0, 0, 0]
        final_move[move] = 1

        reward, done, score = game.play_step(final_move)

        if done:
            # renvoyer le score au lieu de print
            pygame.time.wait(500)
            return score
        

if __name__ == '__main__':
    n_games = 20
    scores = []
    for i in range(n_games):
        score = test()
        print(f"Partie {i+1}/{n_games} terminée — Score : {score}")
        scores.append(score)
    moyenne = sum(scores) / n_games
    print("\n=== Résultats sur 20 parties ===")
    print("Scores :", scores)
    print(f"Score moyen : {moyenne:.2f}")

