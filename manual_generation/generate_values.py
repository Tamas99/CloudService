from Engine import Engine
import pandas as pd
import numpy as np

def main():
    series = pd.Series(dtype=object)
    df = pd.DataFrame({})
    for i in range(10):
        engine = Engine()
        temperatures = np.array(engine.temperature)
        power = np.array(engine.power)
        for j in range(len(temperatures)):
            array = np.array([str(engine.ip_address), int(engine.nr_of_cycles), temperatures[j], power[j]])
            series = pd.Series(array, index=["ip_address", "nr_of_cycles", "temperature", "power"])
            df = df.append(series, ignore_index=True)

    df.to_csv("engines.csv", index=False, header=True)

if __name__ == "__main__":
    main()
