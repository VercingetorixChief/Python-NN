import pygame, sys
import numpy as np
import jumper_agent as agent
from pygame.locals import *


# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

GAME_HEIGHT = 200
GAME_WIDTH = 600

panels = []

class Player:
    def __init__(self, x, y):
        self.position = [x , y]
        self.velocity = [0, 0]

        self.height = 10
        self.width = 10
        
        self.gravity = 0.98
        self.gravity_speed = 0
        self.jumping = True
        self.jump_force = -500

        self.alive = True
        self.friction = 0.995
    
    def reset(self, x, y):
        self.position = [x , y]
        self.velocity = [0, 0]

        self.height = 10
        self.width = 10
        
        self.gravity_speed = 0
        self.jumping = True

        self.alive = True

    def draw(self, surface):
        pygame.draw.rect(surface, BLUE, (self.position[0], self.position[1], self.width, self.height))

    def jump(self):
        if not self.jumping:
            self.velocity[1] += self.jump_force
            self.jumping = True
    
    def move(self, direction):
        self.velocity[0] = direction
        

    def tick(self, collidables, delta):
        self.position[0] += self.velocity[0] * delta
        self.position[1] += self.velocity[1] * delta
        self.apply_gravity(delta)
        self.collision(collidables)

        self.velocity[0] *= self.friction


    
    def apply_gravity(self, delta):
        self.velocity[1] = self.velocity[1] + self.gravity
        

    def collision(self, collidables):
        collision = False

        for collidable in collidables:
            if (self.position[0] < collidable.x + collidable.width and self.position[0] + self.width > collidable.x and self.position[1] < collidable.y + collidable.height and self.position[1] + self.height > collidable.y):
                rock_bottom = collidable.y - self.height
                self.position[1] = rock_bottom
                collision = True
                self.alive = False

        rock_bottom = GAME_HEIGHT - self.height

        if self.position[1] > rock_bottom:
            self.position[1] = rock_bottom
            
            collision = True

            if self.velocity[1] > 0:
                self.velocity[1] = 0
        
        if self.position[0] < 0:
            self.position[0] = 0
            
            self.velocity[0] = 0

        elif self.position[0] > GAME_WIDTH:
            self.position[0] = GAME_WIDTH
            
            self.velocity[0] = 0
        

        if collision:
            if self.jumping:
                self.jumping = False
        
        
        


class Panel:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def move_panel(self, delta):
        self.x -= 200 * delta

    def draw(self, surface):
        pygame.draw.rect(surface, RED, (self.x, self.y, self.width, self.height))
        

class Game:
    def __init__(self, width, height):
        self.panels = []
        self.player = Player(GAME_WIDTH/2, GAME_HEIGHT/2)
        self.next_panel_time = 200
        self.width = width
        self.height = height
        self.surface = pygame.display.set_mode((self.width, self.height), 0, 32)
        self.create_panels()

    def create_panels(self, num = 1):
        self.panels.append(Panel(self.width, self.height - 25, 10, 25))

    def reset(self):
        self.panels = []
        self.next_panel_time = 200
        self.player.reset(GAME_WIDTH/2, GAME_HEIGHT/2)
        self.create_panels()

    def input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.player.jump()
            if event.key == pygame.K_a:
                self.player.move(-100)
            if event.key == pygame.K_d:
                self.player.move(100)

    def tick(self, delta):
        self.next_panel_time -= 100 * delta
        if self.next_panel_time < 0:
            self.create_panels()
            self.next_panel_time = np.random.randint(low = 100, high = 250)

        for panel in self.panels:
            panel.move_panel(delta) 
        
        if len(self.panels) > 0 and self.panels[0].x < -10:
            self.panels.pop(0)
        
        self.player.tick(self.panels, delta)


    def draw(self):
        self.surface.fill(WHITE)

        for panel in self.panels:
            panel.draw(self.surface)

        self.player.draw(self.surface)
    
    def get_closest_panel(self):
        closest = self.panels[0]
        for panel in panels:
            if (panel.x - self.player.position[0] + panel.y - self.player.position[1]) < (closest.x - self.player.position[0] + closest.y - self.player.position[1]) :
                closest = panel

        return closest.x, closest.y

    def get_state(self):
        closest_panel_x, closest_panel_y = self.get_closest_panel()
        return [[closest_panel_x, closest_panel_y , self.player.position[0], self.player.position[1], int(self.player.jumping), self.player.velocity[0], self.player.velocity[1]], not self.player.alive]
    

# set up pygame
pygame.init()

game = Game(GAME_WIDTH, GAME_HEIGHT)


# set up the window
pygame.display.set_caption('Hello world!')


# set up fonts
basicFont = pygame.font.SysFont(None, 48)

ai = agent.Agent()

observation, done = game.get_state()
previous_observation = None

get_ticks_last_frame = 0
current = 0
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


    if t % 50 == 0:
        current += 1
        #print(current)
        action = ai.get_action(observation)
        if action == 1:
            game.player.jump()
        if action == 2:
            game.player.move(-400)
        if action == 3:
            game.player.move(400)

        previous_observation = observation
        observation, done = game.get_state()
        ai.remember(done, action, observation, previous_observation)
        ai.train()


    game.draw()

    if not game.player.alive:
        done = True
        ai.remember(done, action, observation, previous_observation)
        ai.train()
        game.reset()

    # draw the window onto the screen
    pygame.display.update()

