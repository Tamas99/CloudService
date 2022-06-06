from base64 import encode
from aiokafka import AIOKafkaProducer
from fastapi import FastAPI
import uvicorn
import confuse
import pandas as pd
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from logger.logger import rootLogger as logger

logger.debug('Set config settings')
# Instantiates config. Confuse searches for a config_default.yaml
config = confuse.Configuration('sender', __name__)
# Add config items from specified file. Relative path values within the
# file are resolved relative to the application's configuration directory.
config.set_file('config.yaml')

app = FastAPI()

def read():
    # Reading file
    logger.debug("Reading test file")
    data = pd.read_csv('../CMaps/test_FD001.txt', sep=" ", header=None)
    return data

# def data_to_str():
#     data = read()
#     for row in range(len(data[0])):
#         for col in 
#     return data_dict

def get_row(row, y):
    data = read()
    time_series = ''
    for col in range(y):
        time_series += str(data.loc[row,col]) + ' '
    return time_series

@app.post('/send', status_code=200)
async def kafka_produce():
    data = read()
    x, y = data.shape
    host = config['kafka']['host']
    port = config['kafka']['port']
    topic = str(config['kafka']['topic'])
    producer = AIOKafkaProducer(bootstrap_servers=f'{host}:{port}')
    # Get cluster layout and initial topic/partition leadership information
    logger.debug('Starting Kafka producer')
    await producer.start()
    logger.debug('Sending message to Kafka')
    try:
        for row in range(x):
            # for col in data:
            time_series = get_row(row, y)
            # Produce message
            await producer.send_and_wait(topic, bytes(time_series.encode('ascii')))
    finally:
        # Wait for all pending messages to be delivered or expire.
        logger.debug('Stoping Kafka producer')
        await producer.stop()

if __name__ == '__main__':
    module = config['server']['module']
    app = config['server']['fastapiObject']
    host = str(config['server']['host'])
    port = str(config['server']['port'])
    logger.debug('Starting server')
    uvicorn.run(
        f'{module}:{app}',
        host=host,
        port=port
    )
