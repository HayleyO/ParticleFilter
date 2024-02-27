import cv2
import numpy as np

from classes.Drone import Drone
from classes.Simulation_Env import Simulation_Env
from classes.Particle_Filter import Particle_Filter

N = 1000
M = 25
actual_position = np.zeros((2,1))

def draw_map(env:Simulation_Env, particles, weights, drone_position):
    image = None
    for index in range(len(particles)):
        x, y = particles[index][0], particles[index][1]
        image = env.draw_circle_at((x,y), radius=int(weights[index]), color=(255,255,255), thickness=-1)

    image = env.draw_circle_at((drone_position[0], drone_position[1]),color=(155, 255, 255), radius=10, thickness=-1) # Draw Drone
    cv2.imshow("Particle Map", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    map_path = "maps\BayMap.png"
    env = Simulation_Env(map_path)
    drone = Drone(image_size=(env.width, env.height))
    pf = Particle_Filter(image_size=(env.width, env.height), N=N)

    actual_position = drone.position
    x_est, y_est = drone.position[0,0], drone.position[1,0]
    observation = env.reference_image((x_est,y_est), reference_size=M)
    weights = pf.weights(observation, env, M)
    draw_map(env, pf.particles, weights, (x_est, y_est))

    # TODO: 
    # 1. Similarity for weights - proper similarity (MAYBE done)
    # 2. Resample given weights - Do this then sanity check that it works
    
    # 3. Update particles given motion
    # 4. Update position given motion





