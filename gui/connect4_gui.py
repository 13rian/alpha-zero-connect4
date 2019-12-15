import logging
import sys

import pygame

from globals import CONST
from game import connect4
import mcts

# logger
logger = logging.getLogger('Gui')


# colors for the board
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)


# define all pixel sizes
SQUARE_SIZE = 100
WIDTH = CONST.BOARD_WIDTH * SQUARE_SIZE
HEIGHT = (CONST.BOARD_HEIGHT + 1) * SQUARE_SIZE
RADIUS = int(SQUARE_SIZE/2 - 5)


class GUI:
    def __init__(self, network=None):
        # initialize pygame
        pygame.init()

        self.network = network
        self.cpuct = 4
        self.board = connect4.BitBoard()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self. notification_font = pygame.font.SysFont("monospace", 75)

        # draw the board
        self.draw_board()



    def execute_game(self):
        """
        executes the connect4 game
        :return:
        """
        while not self.board.terminal:
            # pygame is event based, it reads all keyboard and mouse events
            for event in pygame.event.get():
                # define the event for the close button
                if event.type == pygame.QUIT:
                    sys.exit()

                # motion bar with the player disk
                if event.type == pygame.MOUSEMOTION:
                    # draw the black rectangle to cover all previous disks
                    pygame.draw.rect(self.screen, BLACK, (0, 0, WIDTH, SQUARE_SIZE))
                    posx = event.pos[0]
                    if self.board.player == CONST.WHITE:
                        pygame.draw.circle(self.screen, RED, (posx, int(SQUARE_SIZE / 2)), RADIUS)
                    else:
                        pygame.draw.circle(self.screen, YELLOW, (posx, int(SQUARE_SIZE / 2)), RADIUS)

                    pygame.display.update()

                # define the click event to set a piece
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # calculate the column from the position of the mouse and play the move
                    posx = event.pos[0]
                    col = int(posx / SQUARE_SIZE)
                    self.board.play_move(col)

                    self.draw_board()       # redraw the board

                    if self.network is not None:
                        self.network_recommendation()


        # show the notification of the player that won
        pygame.draw.rect(self.screen, BLACK, (0, 0, WIDTH, SQUARE_SIZE))
        if self.board.score == 0:
            label = self.notification_font.render("Game drawn!", 1, ORANGE)
        elif self.board.player == CONST.WHITE:
            label = self.notification_font.render("Yellow wins!", 1, YELLOW)
        else:
            label = self.notification_font.render("Red wins!", 1, RED)

        self.screen.blit(label, (40, 10))
        pygame.display.update()  # update the display

        pygame.time.wait(3000)



    def draw_board(self):
        """
        draws the board with the pygame graphics
        :return:
        """
        board_matrix = self.board.to_board_matrix()

        for r in range(CONST.BOARD_HEIGHT):
            for c in range(CONST.BOARD_WIDTH):
                # draw the blue rectangles which define the background
                pygame.draw.rect(
                    self.screen,
                    BLUE,
                    (c * SQUARE_SIZE, r * SQUARE_SIZE + SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                # black circle for empty squares
                if board_matrix[r][c] == 0:
                    pygame.draw.circle(
                        self.screen,
                        BLACK,
                        (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + 1.5 * SQUARE_SIZE)),
                        RADIUS)

                # red circle for player 1
                elif board_matrix[r][c] == 1:
                    pygame.draw.circle(
                        self.screen,
                        RED,
                        (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + 1.5 * SQUARE_SIZE)),
                        RADIUS)

                # yellow circle for player 2
                elif board_matrix[r][c] == 2:
                    pygame.draw.circle(
                        self.screen,
                        YELLOW,
                        (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + 1.5 * SQUARE_SIZE)),
                        RADIUS)

        pygame.display.update()  # update the display


    def play_move(self, col):
        """
        plays the passed move on the board if it is a legal move
        :param col:     column to play
        :return:
        """
        move_legal = self.board.is_legal_move(col)
        if move_legal:
            self.board.play_move(col)
        else:
            logger.debug("ignore illegal move {}".format(col))


    def network_recommendation(self):
        """
        calculates the network policy with mcts
        :return:
        """
        policy = mcts.mcts_policy(self.board, 200, self.network, 1, 0)
        logger.info("network policy: ")
        print(policy)
        print(" ")
