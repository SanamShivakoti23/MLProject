from sensor.exception import SensorException
import os
import sys
from sensor.logger import logging
from sensor.utils import dump_csv_file_to_mongodb_collection
from sensor.constant.database import DATABASE_NAME, COLLECTION_NAME
from sensor.constant.env_variables import FILE_PATH
from dotenv import load_dotenv


# def test_exception():
#     try:
#         logging.info("Division by Zero")
#         a = 1/0
#     except Exception as e:
#         raise SensorException(e, sys)


if __name__ == "__main__":

    # try:
    #     test_exception()
    # except Exception as e:
    #     print(e)
    file_path = os.getenv(FILE_PATH)
    database_name = DATABASE_NAME
    collection_name = COLLECTION_NAME
    dump_csv_file_to_mongodb_collection(
        file_path, database_name, collection_name)
