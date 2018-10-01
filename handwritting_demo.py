from utils import *
import random

data_loader = DataLoader()
for i in range(5):
  draw_strokes(random.choice(data_loader.raw_data))