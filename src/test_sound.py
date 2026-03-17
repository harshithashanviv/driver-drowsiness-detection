import pygame

pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")  # put your alarm file here
pygame.mixer.music.play()
input(" Press Enter to stop.")