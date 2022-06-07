from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI, HTTPException, Response
import uvicorn
import confuse
import pandas as pd
import sys
import os
import pickle
from matplotlib import pyplot as plt
from logger.logger import rootLogger as logger
from elasticsearch import AsyncElasticsearch
from elasticsearch.client import IndicesClient
from elasticsearch.helpers import async_bulk
from datetime import datetime

logger.debug('Set config settings')
# Instantiates config. Confuse searches for a config_default.yaml
config = confuse.Configuration('sender', __name__)
# Add config items from specified file. Relative path values within the
# file are resolved relative to the application's configuration directory.
config.set_file('config.yaml')

app = FastAPI()
elastic_host = config["elasticsearch"]["host"]["prod"].get()

# Create an Asynchronous Elasticsearch object
es = AsyncElasticsearch(
    hosts=[{
        "host": elastic_host,
        "port": config["elasticsearch"]["port"].get(),
        'scheme': config["elasticsearch"]["scheme"].get()
    }],
)

# This gets called once the app is shutting down.
@app.on_event("shutdown")
async def app_shutdown():
    await es.close()

async def create_document(RUL, predictedRUL, file_nr):
    unit_number = "unit_number" + str(file_nr)
    rul = "rul" + str(file_nr)
    predicted = "predicter" + str(file_nr)
    for index in range(len(RUL)):
        yield {
            "_index": 'rul-elastic',
            "doc": {
                unit_number: index+1,
                rul: RUL.loc[index]['RUL'],
                predicted: predictedRUL[index],
                "@timestamp": datetime.now()
            },
        }

def convert_to_df(msgs):
    df = pd.DataFrame({})
    for msg in msgs:
        splitted = msg.split(' ')
        df = df.append(pd.DataFrame(splitted).T)
    return df

@app.post('/consume', status_code=200)
async def kafka_consume():
    host = config['kafka']['host'].get()
    port = config['kafka']['port'].get()
    topic = str(config['kafka']['topic'].get())
    recieved_msgs = []
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
            if msg.value == b'END':
                break
            elif msg.value != b'':
                recieved_msgs.append(msg.value.decode('ascii'))
    finally:
        # Will leave consumer group; perform autocommit if enabled.
        await consumer.stop()
    global data
    data = convert_to_df(recieved_msgs)

def totcycles(data):
    return(data['time_in_cycles'] / (1-data['score']))

def RULfunction(data):
    return(data['maxpredcycles'] - data['max_time_in_cycles'])

def plot_ruls(RUL, predictedRUL):
    plt.figure(figsize = (16, 8))
    plt.plot(RUL, color='red')
    plt.plot(predictedRUL, color='blue')
    plt.xlabel('# Unit', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('RUL', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(['True RUL','Predicted RUL'], bbox_to_anchor=(0., 1.02, 1., .102),
    loc=3, mode='expand', borderaxespad=0)
    plt.show()

@app.post('/predict/{file_nr}', status_code=200)
async def predict(file_nr: int):
    model_file = open('fw_model', 'rb')
    scaler_file = open('fw_scaler', 'rb')
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)
    RUL = pd.read_csv('CMaps/RUL_FD00' + str(file_nr) + '.txt', sep=" ", header=None)
    RUL.drop(columns=[1], inplace=True)
    RUL.columns = ['RUL']
    global data
    test = data.copy()
    test.drop(columns=test.columns[26:],inplace=True)
    columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
            'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]

    test.columns = columns
    #delete columns with constant values ​​that do not carry information about the state of the unit
    test.drop(columns=['Nf_dmd','PCNfR_dmd','P2','T2','TRA','farB','epr'],inplace=True)
    test = test.astype('float64')
    test = test.astype({'unit_number':int, 'time_in_cycles': int})
    ntest = test.copy()
    ntest.iloc[:,2:19] = scaler.transform(ntest.iloc[:,2:19])
    X_test = ntest.values[:,1:19]
    score = model.predict(X_test)
    test = pd.merge(test, test.groupby('unit_number',
        as_index=False)['time_in_cycles'].max(),
        how='left', on='unit_number')
    test.rename(columns={"time_in_cycles_x": "time_in_cycles",
                        "time_in_cycles_y": "max_time_in_cycles"}, inplace=True)
    test['score'] = score
    test = test.T.drop_duplicates().T
    test['maxpredcycles'] = totcycles(test)
    test['RUL'] = RULfunction(test)
    test = test.astype({'unit_number':int, 'time_in_cycles': int})
    t = test.columns == 'RUL'
    ind = [i for i, x in enumerate(t) if x]
    predictedRUL = []
    index_out = -1
    try:
        for i in range(test.unit_number.min(), test.unit_number.max()+1):
            npredictedRUL=test[test.unit_number==i].iloc[test[test.unit_number==i].time_in_cycles.max()-1,ind]
            predictedRUL.append(npredictedRUL)
            index_out = i
    except Exception as e:
        logger.error('Exception: ' + str(e))
        logger.debug('Info - i:{0}, len:{1}, ind:{2}'.format(index_out, len(test.unit_number), ind))
        print(test)
    # plot_ruls(RUL, predictedRUL)
    # Insert data into Elasticsearch
    await async_bulk(es, create_document(RUL, predictedRUL, file_nr))
    return {"result": "Created"}

if __name__ == '__main__':
    module = config['server']['module'].get()
    app = config['server']['fastapiObject'].get()
    host = str(config['server']['host'].get())
    port = str(config['server']['port'].get())
    logger.debug('Starting server')
    uvicorn.run(
        f'{module}:{app}',
        host=host,
        port=port
    )
