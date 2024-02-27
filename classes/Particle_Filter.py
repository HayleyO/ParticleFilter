import cv2
import numpy as np
from classes.Simulation_Env import Simulation_Env
from skimage.metrics import structural_similarity

class Particle_Filter():

    def __init__(self, image_size, N) -> None:
        self.image_size = image_size
        self.N = N
        self.particles = self.generate_initial_particles()

    def generate_initial_particles(self):
        x = np.random.randint(0, self.image_size[0], self.N)
        y = np.random.randint(0, self.image_size[1], self.N)
        return np.column_stack((x,y))

    def get_particle_references(self, env:Simulation_Env, M:int):
        references = []
        for particle in self.particles:
            x_ref,y_ref = particle[0], particle[1]
            reference = env.reference_image((x_ref,y_ref), reference_size=M)
            references.append(reference)
        return references

    def weights(self, observation, env:Simulation_Env, M:int):
        observation = np.float32(observation)
        observation_gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) 
        obs_hist = cv2.calcHist([observation], channels=[0, 1, 2], mask=None, histSize=[8, 8, 8], ranges=[0, 256, 0, 256, 0, 256])
        references = self.get_particle_references(env, M)
        weights = []
        for reference in references:
            reference = np.float32(reference)
            ref_hist = cv2.calcHist([reference], [0,1,2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            similarity = cv2.compareHist(obs_hist, ref_hist, method=cv2.HISTCMP_INTERSECT)
            
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) 
            sim_score, _ = structural_similarity(observation_gray, reference_gray, data_range=256, full=True)
            similarity += (sim_score*100)
            weights.append(similarity)
        #weights = np.divide(weights, np.sum(weights)) # Normalization
        lower_bound = 1
        upper_bound = 25
        weights = [lower_bound + (x - min(weights)) * (upper_bound - lower_bound) / (max(weights) - min(weights)) for x in weights]
        return weights

    def weighted_importance_sampling(self, weights):
        pass

    def move_particles(self, movement):
        pass
