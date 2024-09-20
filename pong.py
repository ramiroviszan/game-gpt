import pygame
import cv2
import numpy as np
import random
import json

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 128, 128
GRID_SIZE = 32 
BALL_RADIUS = 5
PADDLE_WIDTH, PADDLE_HEIGHT = 5, 32
FPS = 240

# Colors
GRID = (25, 100, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")

# Clock to control the frame rate
clock = pygame.time.Clock()

# Initial positions
ball_x, ball_y = WIDTH // 2, HEIGHT // 2

game_speed = 1
ball_speed_x, ball_speed_y = 3 * game_speed, 3 * game_speed

left_paddle_y = (HEIGHT - PADDLE_HEIGHT) // 2
right_paddle_y = (HEIGHT - PADDLE_HEIGHT) // 2
paddle_speed = 5 * game_speed

# AI parameters
ai_speed = 5 * game_speed

left_paddle_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
right_paddle_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
ball_color = (255, 255, 255)

moves = {}

# Main game loop
running = True
tick = 0
strategy = "random"
turn = 'left'
scoreA = 0
scoreB = 0
fontA = pygame.font.Font(None, 20) 
fontB = pygame.font.Font(None, 20) 

# Initialize strategies
left_strategy = "follow" 
right_strategy = "follow" 
ball_served = False

while running and tick <= 1000000:
    print(tick)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the left paddle based on keys
    keys = pygame.key.get_pressed()

    if keys[pygame.K_q]:
        moves[tick-1] = "N"
        moves[tick] = "Q"
        running = False
    else:
        # If the ball is served, update the strategies
        if not ball_served:
            left_strategy = random.choice(["follow", "follow", "follow", "follow", "follow", "follow", "follow", "follow", "follow", "left", "right", "none", "random"])
            right_strategy = random.choice(["follow", "follow", "follow", "follow", "follow", "follow", "follow", "follow", "follow", "left", "right", "none", "random"])
            ball_served = True

        if tick == 0:
            pass
        elif tick == 1:
            moves[tick-1] = "R"
        elif tick == 2:
            moves[tick-1] = "L"
        elif tick == 3:
            moves[tick-1] = "N"
        elif left_strategy == "follow":
            if left_paddle_y + PADDLE_HEIGHT // 2 < ball_y:
                left_paddle_y += paddle_speed
                moves[tick-1] = "R"
            elif left_paddle_y + PADDLE_HEIGHT // 2 > ball_y:
                left_paddle_y -= paddle_speed
                moves[tick-1] = "L"
            else: 
                moves[tick-1] = "N"
        elif left_strategy == "left":
            random_action = "L"
            moves[tick-1] = random_action
            if left_paddle_y > 0:
                left_paddle_y -= paddle_speed
        elif left_strategy == "right":
            random_action = "R"
            moves[tick-1] = random_action
            if left_paddle_y < HEIGHT - PADDLE_HEIGHT:
                left_paddle_y += paddle_speed
        elif left_strategy == "none":
            moves[tick-1] = "N"
        else:
            # Move the left paddle based on a random key strategy
            random_action = random.choice(["L", "R", "N"])  # U: Up, D: Down, N: No movement
            moves[tick-1] = random_action
            if random_action == "L" and left_paddle_y > 0:
                left_paddle_y -= paddle_speed
            elif random_action == "R" and left_paddle_y < HEIGHT - PADDLE_HEIGHT:
                left_paddle_y += paddle_speed
            
        # AI controls the right paddle
        if right_strategy == "follow":
            if right_paddle_y + PADDLE_HEIGHT // 2 < ball_y:
                right_paddle_y += ai_speed
            elif right_paddle_y + PADDLE_HEIGHT // 2 > ball_y:
                right_paddle_y -= ai_speed
        elif right_strategy == "left":
            if right_paddle_y > 0:
                right_paddle_y -= ai_speed
        elif right_strategy == "right":
            if right_paddle_y < HEIGHT - PADDLE_HEIGHT:
                right_paddle_y += ai_speed
        else:
            # Move the right paddle based on a random key strategy
            random_action = random.choice(["L", "R", "N"])
            if random_action == "L" and right_paddle_y > 0:
                right_paddle_y -= ai_speed
            elif random_action == "R" and right_paddle_y < HEIGHT - PADDLE_HEIGHT:
                right_paddle_y += ai_speed

        # Update ball position
        ball_x += ball_speed_x
        ball_y += ball_speed_y

        # Bounce off walls
        if ball_y - BALL_RADIUS <= 0 or ball_y + BALL_RADIUS >= HEIGHT:
            ball_speed_y = -ball_speed_y

        # Bounce off paddles
        if (
            ball_x - BALL_RADIUS <= PADDLE_WIDTH
            and left_paddle_y <= ball_y <= left_paddle_y + PADDLE_HEIGHT
        ): 
            ball_speed_x = -ball_speed_x
            ball_served = False  # Ball is served, update strategies next turn
            left_paddle_color = (random.randint(125, 255), random.randint(125, 255), random.randint(125, 255))
        if (
            ball_x + BALL_RADIUS >= WIDTH - PADDLE_WIDTH
            and right_paddle_y <= ball_y <= right_paddle_y + PADDLE_HEIGHT
        ):
            ball_speed_x = -ball_speed_x
            ball_served = False  # Ball is served, update strategies next turn
            right_paddle_color = (random.randint(125, 255), random.randint(125, 255), random.randint(125, 255))

        # Score points and reposition the ball
        if ball_x - BALL_RADIUS <= 0:
            ball_x, ball_y = WIDTH // 2, HEIGHT // 2
            scoreA += 1
            ball_served = False
        elif ball_x + BALL_RADIUS >= WIDTH:
            ball_x, ball_y = WIDTH // 2, HEIGHT // 2
            scoreB += 1
            ball_served = False

        if scoreA == 50 or scoreB == 50:
            scoreA = 0
            scoreB = 0

    # Draw everything
    screen.fill(BLACK)
    
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRID, (x, 0), (x, HEIGHT), 2)

    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRID, (0, y), (WIDTH, y), 2)

    textA = fontA.render(str(scoreA), True, (255, 0, 0))
    textB = fontB.render(str(scoreB), True, (255, 0, 0))
    screen.blit(textA, (80, 20))
    screen.blit(textB, (40, 20))
    pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), BALL_RADIUS)
    pygame.draw.rect(screen, left_paddle_color, (0, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(
        screen,
        right_paddle_color,
        (WIDTH - PADDLE_WIDTH, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT),
    )

    # Update the display
    pygame.display.flip()

    # Capture the frame
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))  # Rotate 90 degrees counter-clockwise
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"frames/frame_{tick}.png", frame)

    # Control the frame rate
    tick = tick + 1
    clock.tick(FPS)
 

with open('moves.json', 'w') as json_file:
    json.dump(moves, json_file)
# Quit Pygame
pygame.quit()

