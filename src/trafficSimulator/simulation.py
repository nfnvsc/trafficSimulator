from .road import Road
from copy import deepcopy
from .vehicle_generator import VehicleGenerator
from .traffic_signal import TrafficSignal
from .learning.Agent import Agent
from time import sleep, time

import numpy as np
import math
import random
from itertools import combinations

from scipy.spatial.distance import cdist

def check_collision(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) < 2

class Simulation:
    def __init__(self, config={}):
        # Set default configuration
        self.set_default_config()

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

        self.metrics = {}

    def set_default_config(self):
        self.t = 0.0            # Time keeping
        self.frame_count = 0    # Frame count keeping
        self.dt = 1/60          # Simulation time step
        self.roads = []         # Array to store roads
        self.generators = []
        self.traffic_signals = []
        self.agents = []
        self.collisions = 0
        self.configs = []

    def create_road(self, start, end):
        road = Road(start, end)
        self.roads.append(road)
        return road

    def create_roads(self, road_list):
        for road in road_list:
            self.create_road(*road)

    def create_gen(self, config={}, init=True):
        if init: 
            self.configs.append(config)
        gen = VehicleGenerator(self, config)
        self.generators.append(gen)
        return gen

    def create_signal(self, roads, config={}):
        roads = [[self.roads[i] for i in road_group] for road_group in roads]

        sig = TrafficSignal(roads, config)
        self.traffic_signals.append(sig)

        agent = Agent(self, sig, {
            "id": len(self.agents),
            "epsilon": 0.4,
            "alpha": 0.6,
            "gamma": 0.9
        })

        self.agents.append(agent)
        return sig

    @property
    def state(self):
        vehicle_count = []
        avg_speed = 0
        total_vehicles = 1
        for r in self.roads:
            if r.has_traffic_signal:
                vehicle_count.append(len(r.vehicles))
                for v in r.vehicles:
                    total_vehicles += 1
                    avg_speed += v.v
        return [[*vehicle_count], int(avg_speed/total_vehicles), self.collisions]

    def update_metrics(self):
        #a = np.array([road.metrics['avg_speed'] for road in self.roads])
        #avg = a[a!=0].mean()

        self.metrics = {
            "collisions": self.collisions,
           # "avg_speed": avg
        }


    def _check_collisions(self):
        vehicles = []
        xs = []
        ys = []
        for i, road in enumerate(self.roads):
            for vehicle in road.vehicles:
                sin, cos = road.angle_sin, road.angle_cos
                x = road.start[0] + cos * vehicle.x 
                y = road.start[1] + sin * vehicle.x 
                vehicles += [(x, y, i)]
                xs.append(x)
                ys.append(y)

        for i, v1 in enumerate(vehicles):
            for i2 in range(i + 1, len(vehicles)):
                v2 = vehicles[i2]
                if v1[2] != v2[2] and check_collision(v1, v2):
                    self.collisions += 1

    def update(self):
        # Update every road
        time = 60
        for road in self.roads:
            road.update(self.dt)

        # Add vehicles
        for gen in self.generators:
            gen.update()
        
        if self.frame_count % (time / self.dt) == 0:
            for agent in self.agents:
                agent.act()

        # Check roads for out of bounds vehicle
        for road in self.roads:
            # If road has no vehicles, continue
            if len(road.vehicles) == 0: continue
            # If not
            vehicle = road.vehicles[0]
            # If first vehicle is out of road bounds
            if vehicle.x >= road.length:
                # If vehicle has a next road
                if vehicle.current_road_index + 1 < len(vehicle.path):
                    # Update current road to next road
                    vehicle.current_road_index += 1
                    # Create a copy and reset some vehicle properties
                    new_vehicle = deepcopy(vehicle)
                    new_vehicle.x = 0
                    # Add it to the next road
                    next_road_index = vehicle.path[vehicle.current_road_index]
                    self.roads[next_road_index].vehicles.append(new_vehicle)
                # In all cases, remove it from its road
                road.vehicles.popleft() 
            
        if self.frame_count % (time / self.dt) == (time / self.dt / 2):
            for agent in self.agents:
                agent.update()

        self._check_collisions()

        # Increment time
        self.t += self.dt
        self.frame_count += 1
        self.update_metrics()

        if self.collisions > 1:
            print(self.state)
            self.reset()


    def run(self, steps):
        for _ in range(steps):
            self.update()
        
    def reset(self):
        self.t = 0.0
        self.frame_count = 0
        self.collisions = 0

        self.generators = []

        for config in self.configs:
            self.create_gen(config, init=False)

        for road in self.roads:
            road.reset()
        
        for agent in self.agents:
            agent.reset()
        