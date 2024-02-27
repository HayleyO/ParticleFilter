import math
import numpy as np

class Drone():

    def __init__(self, image_size) -> None:
        self.image_size = image_size
        self.position = self.random_starting_location()

    def random_starting_location(self):
        '''Returns a numpy 2 x 1 array [[x], [y]]'''
        x = int(np.random.uniform(0, self.image_size[0]))
        y = int(np.random.uniform(0, self.image_size[1]))
        return np.array([[x],[y]])
    
    def random_movement(self):
        # Randomly generate movement that squares add up to 1
        dx = np.random.uniform(0, 1)
        dy = math.sqrt(1.0 - dx**2)
        # Make the movement work in opposite directions
        if np.random.uniform(0, 1) > 0.5:
            dx = -dx
        if np.random.uniform(0, 1) > 0.5:
            dy = -dy
        movement = np.array([[dx],[dy]])
        # Check if the movement is expected to be outside the map -- since the drone doesn't know actual position we will need to do this somewhere else, too
        if (self.position + movement)[0,0] > self.image_size[0] or (self.position + movement)[1,0] > self.image_size[1] or (self.position + movement)[0,0] < 0 or (self.position + movement)[1,0] < 0:
            print("ERROR: Expected movement outside map!")
            return None
        return movement
    
    def update_position_estimate(self, movement):
        self.position = np.add(self.position, movement)
