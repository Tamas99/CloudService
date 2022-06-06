import logging
import os

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

# Create logs dir if not exists
path = 'logs'
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
    print('not')
    # Create a new directory because it does not exist 
    os.makedirs(path)

fileHandler = logging.FileHandler("{0}/{1}.log".format('logs', 'logs'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
