from multiprocessing import cpu_count, Process, Lock, Manager
from multiprocessing.managers import BaseManager, DictProxy, ListProxy

from src.trafficSimulator.simulation import Simulation
from collections import defaultdict

import pickle
import time

class MultithreadSimulation:
    def __init__(self, config={}) -> None:
        self.processes = []
        self.manager = Manager()
        self.shared = self.manager.list()
        self.simulations = []

        config.update({
            "multithreaded": True,
            "shared": self.shared,
        })

        for _ in range(cpu_count()):
            sim = Simulation(config)
            self.simulations.append(sim)
        
    
    def create_roads(self, road_list):
        for sim in self.simulations:
            sim.create_roads(road_list)

    def create_gen(self, config={}):
        for sim in self.simulations:
            sim.create_gen(config)

    def create_signal(self, roads, config={}):
        lock = Lock()
        self.shared.append((self.manager.dict(self.load_shared(len(self.shared)))))
        for sim in self.simulations:
            sim.create_signal(roads, lock, config) 

    def load_shared(self, id):
        try:
            with open(f'qtable{id}.pickle', 'rb') as file:
                q_table = pickle.load(file)
        except FileNotFoundError:
            q_table = {}
        
        return q_table

    def save_shared(self):
        while True:
            for id, s in enumerate(self.shared):
                with open(f'qtable{id}.pickle', 'wb') as file:
                    pickle.dump(dict(s), file)
            time.sleep(5)
            

    def run(self):
        for sim in self.simulations:
            p = Process(target=sim.run_forever)
            p.start()
            self.processes.append(p)

        p = Process(target=self.save_shared)
        p.start()
        self.processes.append(p)
        
        for p in self.processes:
            p.join()
        


    