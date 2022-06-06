from aiokafka import AIOKafkaConsumer
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

@app.post('/consume', status_code=200)
async def kafka_consume():
    host = config['kafka']['host']
    port = config['kafka']['port']
    topic = str(config['kafka']['topic'])
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=f'{host}:{port}',
        group_id="group")
    # Get cluster layout and join group `group`
    await consumer.start()
    try:
        # Consume messages
        async for msg in consumer:
            print("consumed: ", msg.topic, msg.key, msg.value)
    finally:
        # Will leave consumer group; perform autocommit if enabled.
        await consumer.stop()

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
