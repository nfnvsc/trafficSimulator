from multiprocessing import cpu_count, Process, Lock, Manager, Queue

from src.trafficSimulator.simulation import Simulation

import pickle
import time


class MultithreadSimulation:
    def __init__(self, config={}) -> None:
        self.processes = []
        self.manager = Manager()
        self.shared = self.manager.list()
        self.shared_metrics = Queue()
        self.simulations = []
        self.current_metrics = []

        config.update({
            "id": 0,
            "multithreaded": True,
            "shared": self.shared,
            "manager": self.manager,
            "shared_metrics": self.shared_metrics
        })

        #for i in range(cpu_count() - 4):
        for i in range(2):
            config["id"] = i
            sim = Simulation(config)
            self.simulations.append(sim)
        
    @property
    def roads(self):
        return self.simulations[0].roads
    
    @property
    def traffic_signals(self):
        return self.simulations[0].traffic_signals

    @property
    def t(self):
        return self.simulations[0].t
    
    @property
    def frame_count(self):
        return self.simulations[0].frame_count
    
    @property
    def metrics(self):
        return self.simulations[0].metrics


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

            time.sleep(20)

    def save_metrics(self):
        while True:
            item = self.shared_metrics.get(True)

            self.current_metrics.append(item)
            print(item)

    def run(self, steps=1):       
        self.simulations[0].run(steps)

    def run_forever(self):
        for sim in self.simulations:
            p = Process(target=sim.run_forever)
            p.start()
            self.processes.append(p)

        p = Process(target=self.save_shared)
        p.start()
        self.processes.append(p)

        p = Process(target=self.save_metrics)
        p.start()
        self.processes.append(p)

        for p in self.processes:
            p.join()
        


    