import pygame, sys
import numpy as np
import math as math
import tictactoe_agent as agent
from pygame.locals import *


# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

GAME_HEIGHT = 200
GAME_WIDTH = 200
PLAYER_WON = 0
results = [0, 0,0,0]

class Board:
    def __init__(self):
        self.board = [0,0,0, 0,0,0, 0,0,0]

    def reset(self):
        self.board = [0,0,0, 0,0,0, 0,0,0]

    def is_move_legal(self, pos):
        if self.board[pos
        ] == 0:
            return True
        else:
            return False

    def make_move(self, pos, player):
        self.board[pos] = player
        return self.check_win(pos, player)

    def check_win(self, pos, player):
        col = pos % 3
        row = math.floor(pos / 3)

        win = [True, True, True]

        for x in range(3):
            if self.board[x + row * 3] != player:
                win[0] = False
        
        for y in range(3):
            if self.board[y * 3 + col] != player:
                win[1] = False
        
        if self.board[4] != player:
            win[2] = False
        
        if(pos == 0):
            if(self.board[8] != player):
                win[2] = False

        if(pos == 2):
            if(self.board[6] != player):
                win[2] = False
        
        if(pos == 6):
            if(self.board[2] != player):
                win[2] = False
         
        if(pos == 8):
            if(self.board[0] != player):
                win[2] = False
            

        if(pos == 4):
            if(self.board[0] != player or self.board[8] != player):
                win[2] = False
            elif(self.board[6] != player or self.board[2] != player):
                win[2] = False   

        if (not (pos == 0 or pos == 2 or pos == 4 or pos == 6 or pos == 8)):
            win[2] = False

        if True in win:
            return player

        if not 0 in self.board:
            return 3
        
        return 0

    def draw(self, surface):

        for i in range(3):
            pygame.draw.line(surface, BLACK, (i * (GAME_WIDTH / 3), 0), (i * (GAME_WIDTH / 3), GAME_HEIGHT) , 2)
            pygame.draw.line(surface, BLACK, (0, i * (GAME_HEIGHT / 3)), (GAME_WIDTH, i * (GAME_HEIGHT / 3)) , 2)
        
        for i in range(len(self.board)):
            if(self.board[i] == 1):
                pygame.draw.rect(surface, GREEN, ( (i % 3) * (GAME_WIDTH / 3) + (((GAME_WIDTH/3) / 2) - 10), math.floor(i / 3) * (GAME_HEIGHT / 3) + (((GAME_HEIGHT/3) / 2) - 10), 20, 20))
            elif(self.board[i] == 2):
                 pygame.draw.rect(surface, RED, ( (i % 3) * (GAME_WIDTH / 3) + (((GAME_WIDTH/3) / 2) - 10), math.floor(i / 3) * (GAME_HEIGHT / 3) + (((GAME_HEIGHT/3) / 2) - 10), 20, 20))
            
            
    
class Game:
    def __init__(self, width, height):
        self.board = Board()
        self.width = width
        self.height = height
        self.surface = pygame.display.set_mode((self.width, self.height), 0, 32)

    def reset(self):
        self.board.reset()

    def input(self, event):
        return

    def tick(self, delta):
        return

    def draw(self):
        self.surface.fill(WHITE)

        self.board.draw(self.surface)

    def get_result(self):
        pass

    def get_state(self):
        return self.board.board
    

# set up pygame
pygame.init()

game = Game(GAME_WIDTH, GAME_HEIGHT)


# set up the window
pygame.display.set_caption('Hello world!')


# set up fonts
basicFont = pygame.font.SysFont(None, 48)

ai1 = agent.Agent(1)
ai2 = agent.Agent(2)

#observation, done = game.get_state()
#previous_observation = None

get_ticks_last_frame = 0
current = 1
last_start = 1
# run the game loop
while True:
    t = pygame.time.get_ticks()
    # deltaTime in seconds.
    delta_time = (t - get_ticks_last_frame) / 1000.0
    get_ticks_last_frame = t
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            game.input(event)

    game.tick(delta_time)


    if t % 10 == 0:
        if current == 1:
            actions = ai1.get_actions(game.get_state())
            probs = np.copy(actions)

            for index in range(len(actions)):
                
                if not game.board.is_move_legal(index):
                    probs[index] = -1

            PLAYER_WON = game.board.make_move(np.argmax(probs), 1)
            ai1.remember(np.argmax(probs), game.get_state())
            current = 2

        else:
            actions = ai2.get_actions(game.get_state())
            probs = np.copy(actions)

            for index in range(len(actions)):
                if not game.board.is_move_legal(index):
                    probs[index] = -1
                    
            PLAYER_WON = game.board.make_move(np.argmax(probs), 2)
            ai2.remember(np.argmax(probs), game.get_state())
            current = 1
        
        game.draw()

    
    if PLAYER_WON > 0:
        
        ai1.train(PLAYER_WON)
        ai2.train(PLAYER_WON)
        game.reset()
        results[PLAYER_WON] += 1
        PLAYER_WON = 0
        if last_start == 1:
            current = 2
            last_start = 2
        else:
            last_start = 1
            current = 1
        print(results)


    # draw the window onto the screen
    pygame.display.update()

