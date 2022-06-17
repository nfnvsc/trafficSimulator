import time

class TrafficSignal:
    def __init__(self, roads, config={}):
        # Initialize roads
        self.roads = roads
        # Set default configuration
        self.set_default_config()
        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)
        # Calculate properties
        self.init_properties()

        self.metrics = {}

    def set_default_config(self):
        self.cycle = [(False, True), (True, False)]
        self.slow_distance = 50
        self.slow_factor = 0.4
        self.stop_distance = 15

        self.current_cycle_index = 0
        self.previous_cycle = None

        self.last_t = 0

    def init_properties(self):
        for i in range(len(self.roads)):
            for road in self.roads[i]:
                road.set_traffic_signal(self, i)

    def update_metrics(self):
        cycle_time = 0
        if self.previous_cycle == None:
            self.previous_cycle = self.current_cycle_index
            self.cycle_time = time.time()
        elif self.previous_cycle != self.current_cycle_index:
            cycle_time = time.time() - self.cycle_time
            self.cycle_time = 0
            self.previous_cycle = self.current_cycle_index

        self.metrics = {"cycle_time": cycle_time}

    @property
    def current_cycle(self):
        return self.cycle[self.current_cycle_index]
    
    @current_cycle.setter
    def current_cycle(self, val):
        self.current_cycle_index = int(val)

    def update(self, sim):
        cycle_length = 15
        k = (sim.t // cycle_length) % 2
        self.current_cycle_index = int(k)

        self.update_metrics()


