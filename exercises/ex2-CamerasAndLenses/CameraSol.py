import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
import math


def ex1():
    a = 10
    b = 3
    # Answer is in radians
    print(math.atan2(b, a))
    print("In deg")
    print(math.atan2(b, a)*180/math.pi)

# ex 2. Gauss length equation, calculated in meters.


def camera_b_distance(f, g):
    return -1*((g*f)/(f-g))


focal = 0.015
print(camera_b_distance(0.005, 5))
