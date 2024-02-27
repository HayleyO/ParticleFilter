import cv2
import numpy as np

class Simulation_Env():

    def __init__(self, image_path) -> None:
        self.image_path = image_path
        self.image = self.load_image()

        self.height = self.image.shape[:2][0]
        self.width = self.image.shape[:2][1]

        self.origin = (self.width//2,self.height//2) # Origin 0,0 is center of the image

    def load_image(self):
        return cv2.imread(self.image_path)
    
    def reference_image(self, reference_position, reference_size):
        x = reference_position[0]
        y = reference_position[1]

        x1, x2 = x-reference_size//2, (x+reference_size//2)
        y1, y2 = y-reference_size//2, (y+reference_size//2)
        # Handle moving towards the edges of map
        top, left = False, False # Makes assumption we only need to pad certain sides
        if x1 < 0:
            x1 = 0
            left = True
        if x2 > self.width-1:
            x2 = self.width-1
        if y1 < 0:
            y1 = 0
            top = True
        if y2 > self.height-1:
            y2 = self.height-1
        image = self.image[y1:y2, x1:x2]
        # If the reference image is slightly smaller because of where the particle is, pad the unseen spaces with -1
        x_difference = reference_size - image.shape[:2][1]
        y_difference = reference_size - image.shape[:2][0]
        if x_difference > 0:
            pad_x = np.ones((image.shape[:2][0], x_difference, 3)) * -256
            if left:
                image = np.append(pad_x, image, axis=1)
            else:
                image = np.append(image, pad_x, axis=1)
        if y_difference > 0:
            pad_y = np.ones((y_difference, image.shape[:2][1], 3)) * -256
            if top:
                image = np.append(pad_y, image, axis=0)
            else:
                image = np.append(image, pad_y, axis=0)
        return image
    
    def draw_circle_at(self, position, radius:int=5, color=(255,0,0), thickness=2):
        return cv2.circle(self.image, position, radius=radius, color=color, thickness=thickness) 

if __name__ == "__main__":
    map_path = "maps\BayMap.png"
    env = Simulation_Env(map_path)
