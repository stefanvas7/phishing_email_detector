import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[ %(asctime)s ] line: %(lineno)d %(name)s -- level: %(levelname)s -- %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level = logging.INFO

)

