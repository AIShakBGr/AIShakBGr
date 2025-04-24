import pygame
import time
import random

# Initialisation de Pygame
pygame.init()

# Dimensions de la fenêtre réduite
window_x = 480
window_y = 360

# Couleurs
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
grey = pygame.Color(128, 128, 128)  # Couleur pour les obstacles

# Vitesse du serpent
snake_speed = 15

# Création de la fenêtre de jeu
pygame.display.set_caption('Snake Game')
game_window = pygame.display.set_mode((window_x, window_y))

# FPS (frames per second) controller
fps = pygame.time.Clock()

# Taille des blocs du jeu
block_size = 10

# Position et corps du serpent
snake_position = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
fruit_position = [random.randrange(1, (window_x // block_size)) * block_size,
                    random.randrange(1, (window_y // block_size)) * block_size]
fruit_spawn = True
direction = 'RIGHT'
change_to = direction
score = 0

# Position et dimensions des obstacles en forme de crochets et barres
obstacle_thickness = block_size
obstacles = [
    # Crochets dans les coins
    pygame.Rect(block_size, block_size, obstacle_thickness, obstacle_thickness * 3),  # Haut gauche (vertical)
    pygame.Rect(block_size, block_size, obstacle_thickness * 3, obstacle_thickness),  # Haut gauche (horizontal)
    pygame.Rect(window_x - 2 * block_size, block_size, obstacle_thickness, obstacle_thickness * 3),  # Haut droit (vertical)
    pygame.Rect(window_x - 4 * block_size, block_size, obstacle_thickness * 3, obstacle_thickness),  # Haut droit (horizontal)
    pygame.Rect(block_size, window_y - 4 * block_size, obstacle_thickness, obstacle_thickness * 3),  # Bas gauche (vertical)
    pygame.Rect(block_size, window_y - 2 * block_size, obstacle_thickness * 3, obstacle_thickness),  # Bas gauche (horizontal)
    pygame.Rect(window_x - 2 * block_size, window_y - 4 * block_size, obstacle_thickness, obstacle_thickness * 3),  # Bas droit (vertical)
    pygame.Rect(window_x - 4 * block_size, window_y - 2 * block_size, obstacle_thickness * 3, obstacle_thickness),  # Bas droit (horizontal)
    # Barres horizontales au milieu
    pygame.Rect(window_x // 4, window_y // 3, window_x // 2, obstacle_thickness),  # Barre du haut
    pygame.Rect(window_x // 4, 2 * window_y // 3, window_x // 2, obstacle_thickness)   # Barre du bas
]

# Fonction pour afficher le score
def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)

# Fonction de fin de jeu
def game_over():
    my_font = pygame.font.SysFont('times new roman', 50)
    game_over_surface = my_font.render('Your Score is : ' + str(score), True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (window_x / 2, window_y / 4)
    game_window.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    time.sleep(2)
    pygame.quit()
    quit()

# Boucle principale du jeu
while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                change_to = 'UP'
            if event.key == pygame.K_DOWN:
                change_to = 'DOWN'
            if event.key == pygame.K_LEFT:
                change_to = 'LEFT'
            if event.key == pygame.K_RIGHT:
                change_to = 'RIGHT'

    # Validation de la direction
    if change_to == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP':
        direction = 'DOWN'
    if change_to == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if change_to == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'

    # Déplacement du serpent
    if direction == 'UP':
        snake_position[1] -= block_size
    if direction == 'DOWN':
        snake_position[1] += block_size
    if direction == 'LEFT':
        snake_position[0] -= block_size
    if direction == 'RIGHT':
        snake_position[0] += block_size

    # Gestion du passage à travers les bords
    if snake_position[0] < 0:
        snake_position[0] = window_x - block_size
    elif snake_position[0] >= window_x:
        snake_position[0] = 0
    if snake_position[1] < 0:
        snake_position[1] = window_y - block_size
    elif snake_position[1] >= window_y:
        snake_position[1] = 0

    # Mécanisme de croissance du serpent
    snake_body.insert(0, list(snake_position))
    if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
        score += 10
        fruit_spawn = False
    else:
        snake_body.pop()

    if not fruit_spawn:
        while True:  # Boucle infinie jusqu'à trouver une position valide
            fruit_position = [random.randrange(1, (window_x // block_size)) * block_size,
                                random.randrange(1, (window_y // block_size)) * block_size]
            is_on_obstacle = False
            fruit_rect = pygame.Rect(fruit_position[0], fruit_position[1], block_size, block_size)
            for obstacle in obstacles:
                if fruit_rect.colliderect(obstacle):
                    is_on_obstacle = True
                    break  # Sortir de la boucle des obstacles si collision
            if not is_on_obstacle:
                break  # Sortir de la boucle while si la position est valide
        fruit_spawn = True

    game_window.fill(black)

    # Dessiner les obstacles
    for obstacle in obstacles:
        pygame.draw.rect(game_window, grey, obstacle)

    for pos in snake_body:
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], block_size, block_size))

    pygame.draw.rect(game_window, white, pygame.Rect(fruit_position[0], fruit_position[1], block_size, block_size))

    # Condition de collision avec les obstacles
    for obstacle in obstacles:
        if pygame.Rect(snake_position[0], snake_position[1], block_size, block_size).colliderect(obstacle):
            game_over()

    # Condition de collision avec le corps
    for block in snake_body[1:]:
        if snake_position[0] == block[0] and snake_position[1] == block[1]:
            game_over()

    show_score(1, white, 'consolas', 20)

    pygame.display.flip()

    fps.tick(snake_speed)