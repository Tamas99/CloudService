import random
import socket
import struct
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Creates a data dir and a filepath with the csv filename
filepath = Path("data/engine_ips.csv")  
filepath.parent.mkdir(parents=True, exist_ok=True) 

def generate_engine_ips(nr_of_engines=10):
    engine_ips = pd.Series([], dtype=str)
    for i in range(nr_of_engines):
        # defines a random ip address
        ip = pd.Series([socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))])
        engine_ips = pd.concat([engine_ips, ip])

    print(len(engine_ips))
    print(np.sum(engine_ips.duplicated() == False)) # prints the number of unique ips
    logging.debug(len(engine_ips))
    logging.debug(np.sum(engine_ips.duplicated() == False))

    engine_ips.drop_duplicates(keep="last")
    # Save to csv file
    engine_ips.to_csv(filepath, index=False, header=False)

if __name__ == "__main__":
    generate_engine_ips()
