import matplotlib.pyplot as plt
import random as r
import numpy as np

from Engine import Engine

# list.append(r.triangular(low=1, high=10, mode=3))
# list.append(r.betavariate(alpha=1, beta=2))
# list.append(r.expovariate(lambd=0.1))
# list.append(r.gammavariate(alpha=1, beta=2))

def custom_temperature_generator(engine: Engine, sample: int = 1000) -> list[float]:
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
    percentage_to_fail = get_percentage(engine.nr_of_cycles)
    # This value will determine if the engine fails or not
    determinant_value = r.uniform(a=0, b=100)
    while len(list) < sample:
        value = r.uniform(a=start, b=end)
        if determinant_value < percentage_to_fail:
            print(determinant_value)
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

def get_percentage(nr):
    if nr < 1000:
        return 5
    elif nr < 1200:
        return 20
    else:
        return 50

engine = Engine(192, 0, 20, 20, 1201)
def generator_analyzing():
    for _ in range(5):
        list = custom_temperature_generator(engine)
        plt.plot(list)
        plt.show()

generator_analyzing()
# custom_temperature_generator(engine)
