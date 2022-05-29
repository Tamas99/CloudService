import matplotlib.pyplot as plt
import random as r
import numpy as np

class Engine():
    def __init__(self, ip_address="", cycles=0):
        # TODO: Parameter checking for invalid values
        self.ip_address = ip_address
        self.nr_of_cycles = cycles
        self.fails_under_pressure = False
        self.sensor = self.generate_temperature()
        self.power = self.generate_engine_power()
    
    def generate_temperature(self, sample: int = 1000) -> list[float]:
        '''
        Custom temperature generator for an engine
        '''
        list = []
        start = 20
        end = 23
        for _ in range(5):
            list.append(r.uniform(a=start, b=end))
        avg = np.average(list)
        list = [list[-1]]
        percentage_to_fail = self.get_percentage(self.nr_of_cycles)
        # This value will determine if the engine fails or not
        determinant_value = r.uniform(a=0, b=100)
        while len(list) < sample:
            value = r.uniform(a=start, b=end)
            if determinant_value < percentage_to_fail:
                self.fails_under_pressure = True
                if value >= 100:
                    while len(list) < sample:
                        list.append(r.uniform(102, 107))
                    break
                elif value >= 20:
                    list.append(value)
            elif value >= 20:
                list.append(value)
            start = start + (list[-1] - avg)
            end = end + (list[-1] - avg)
            list_for_avg = [start, end]
            list_for_avg.append(list[-1])
            try:
                list_for_avg.append(list[-2])
            except:
                pass
            avg = np.average(list_for_avg)
        
        # plt.plot(list)
        # plt.show()
        return list

    def get_percentage(self, nr_of_cycles):
        if nr_of_cycles > 1200:
            return 50
        elif nr_of_cycles > 1000:
            return 20
        else:
            return 5

    def generate_engine_power(self):
        list = []
        start = 0
        end = 5
        failed = False
        for _ in range(10):
            list.append(r.uniform(a=start, b=end))
        avg = np.average(list)
        index = 0
        while len(list) < len(self.sensor):
            if self.sensor[index] > 100 and self.fails_under_pressure == True:
                failed = True
                break
            else:
                value = r.uniform(a=start, b=end)
                if value > list[-1] - 2 and value < 60 and value > 0:
                    list.append(value)
                    index = index + 1
                    start = start + (list[-1] - avg)
                    end = end + (list[-1] - avg)
                    list_for_avg = [start, end]
                    list_for_avg.append(list[-1])
                    try:
                        list_for_avg.append(list[-2])
                    except:
                        pass
                    avg = np.average(list_for_avg)
                elif value >= 60:
                    start = 57
                    end = 63
                    list.append(value)
                    index = index + 1

            
        # Decreasing engine power to 0 after failure
        if failed == True:
            while len(list) < len(self.sensor):
                if list[-1] > 3:
                    value = r.uniform(a=1, b=3)
                    list.append(list[-1] - value)
                else:
                    list.append(0)
        return list

engine = Engine(192, 1201)
plt.plot(engine.sensor)
plt.plot(engine.generate_engine_power())
plt.show()
