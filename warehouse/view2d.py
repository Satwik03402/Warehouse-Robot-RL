"""Pygame visualization of the warehouse and an A* robot replay."""
import pygame
from .layout import LAYOUT

CELL = 48

def play(path, fps=5):
    pygame.init()
    w, h = LAYOUT.width * CELL, LAYOUT.height * CELL
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption('Warehouse Navigation | 2D baseline')
    clock = pygame.time.Clock()
    index, active = 0, True
    def box(cell):
        x, y = cell
        return pygame.Rect(x * CELL, (LAYOUT.height - 1 - y) * CELL, CELL, CELL)
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                active = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                active = False
        screen.fill((237, 243, 249))
        for x in range(LAYOUT.width):
            for y in range(LAYOUT.height):
                pygame.draw.rect(screen, (211, 221, 232), box((x, y)), 1)
        for cell in LAYOUT.shelves:
            pygame.draw.rect(screen, (96, 112, 132), box(cell).inflate(-4, -4), border_radius=5)
        pygame.draw.rect(screen, (248, 185, 75), box(LAYOUT.start).inflate(-12, -12), border_radius=6)
        pygame.draw.rect(screen, (53, 181, 109), box(LAYOUT.goal).inflate(-12, -12), border_radius=6)
        if len(path) > 1:
            centers = [box(c).center for c in path]
            pygame.draw.lines(screen, (55, 162, 213), False, centers, 3)
        pygame.draw.circle(screen, (32, 90, 211), box(path[index]).center, CELL // 3)
        pygame.display.flip()
        if index < len(path) - 1:
            index += 1
        clock.tick(fps)
    pygame.quit()
