from sensor.exception import SensorException
import os
import sys
from sensor.logger import logging
from sensor.utils2 import dump_csv_file_to_mongodb_collection
from sensor.constant.database import DATABASE_NAME, COLLECTION_NAME
from sensor.constant.env_variables import FILE_PATH
from dotenv import load_dotenv

from sensor.pipeline.training_pipeline import TrainPipeline

from sensor.utils.main_utils import load_object
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os
import sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
import os
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SAVED_MODEL_DIR


from fastapi import FastAPI
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, File, UploadFile, Response
import pandas as pd
import numpy as np
import uvicorn
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH, TARGET_COLUMN
from sensor.utils.main_utils import load_numpy_array_data
# def test_exception():
#     try:
#         logging.info("Division by Zero")
#         a = 1/0
#     except Exception as e:
#         raise SensorException(e, sys)


app = FastAPI()


origins = ["*"]
# Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train():
    try:

        training_pipeline = TrainPipeline()

        if training_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")

        training_pipeline.run_pipeline()
        return Response("Training successfully completed!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/predict")
async def predict():
    try:

        # get data and from the csv file
        # covert it into dataframe

        load_dotenv()
        file_path = os.getenv(FILE_PATH)

        # Check if file_path is None or invalid
        if file_path is None or not os.path.exists(file_path):
            return Response("Error: CSV file not found or path is invalid.")

        df = pd.read_csv(file_path)

        # Ensure `df` is not empty
        if df is None or df.empty:
            return Response("Error: The CSV file is empty or could not be loaded.")

        schema_path = read_yaml_file(SCHEMA_FILE_PATH)
        columns_to_drop = schema_path["drop_columns"]
        df = df.drop(columns_to_drop, axis=1)
        df.replace({"na": 0}, inplace=True)
        print(df[TARGET_COLUMN])
        # df[TARGET_COLUMN].replace({"pos": 1, "neg": 0}, inplace=True)
        df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())
        print(TargetValueMapping().to_dict())
        df_array = df.to_numpy()
        X = df_array[:, :-1]
        y = df_array[:, -1]

        Model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not Model_resolver.is_model_exists():
            return Response("Model is not available")

        best_model_path = Model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        y_pred = model.predict(X)
        df['predicted_column'] = y_pred
        df['predicted_column'] = df['predicted_column'].replace(
            TargetValueMapping().reverse_mapping())
        # df['predicted_column'] = df['predicted_column'].map(
        #     {0: 'negative', 1: 'positive'})

        prediction = df['predicted_column'].iloc[0]

        # get the prediction as output
        return prediction

    except Exception as e:
        raise SensorException(e, sys)


def main():
    try:

        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)


if __name__ == "__main__":

    # try:
    #     test_exception()
    # except Exception as e:
    #     print(e)
    # load_dotenv()
    # file_path = os.getenv(FILE_PATH)
    # database_name = DATABASE_NAME
    # collection_name = COLLECTION_NAME
    # dump_csv_file_to_mongodb_collection(
    #     file_path, database_name, collection_name)
    # app_run(app, host=APP_HOST, port=APP_PORT)
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
