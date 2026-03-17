import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")
pygame.mixer.music.play()

print("Playing alarm...")
time.sleep(5)

pygame.mixer.music.stop()
print("Stopped")