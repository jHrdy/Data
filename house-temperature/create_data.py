# script for generating random data for testing

import numpy as np
import matplotlib.pyplot as plt
from random import randint, seed
import pandas as pd

seed(42)

# noise to create "more real" data
noise_for_room = np.random.normal(size=24)

def add_noise(x):
    if randint(-5,5) % 2 == 0:
        return -0.9*x**2-0.05*x+1.3
    return 0.9*x**2-0.05*x+1.3

# creating data
temperature = [20 + add_noise(randint(-150,150)/100) for i in range(24)]
idx = [i for i in range(len(temperature))]

# functions to create data for the places in house each set to specific mean value
# using different approach and parameter values to simulate real life conditions 
# data have various noise dispersion resulting in specific value fluctuations for each place 
# hall temperature generated with dependency to room and kitchen pseudo-simulating real house

def random_room():
    return [20.4 + add_noise(randint(-150,150)/100) for i in range(24)]

def random_kitchen():
    return [20.6 + add_noise(randint(-100,200)/100) for i in range(24)]

def random_hall():
    hall = []
    for i in range(24):
        data = (random_room()[i] + random_kitchen()[i] + add_noise(randint(-50,75)/100))/2
        hall.append(data)
    return hall

kitchen_data = []
room_data = []
hall_data = []

# function to be called from code where data are needed
def data():
    for i in range(7):
        kitchen = random_kitchen()
        kitchen_data.append(kitchen)
        room = random_room()
        room_data.append(room)
        hall = random_hall()
        hall_data.append(hall)
        
    return kitchen_data, room_data, hall_data



