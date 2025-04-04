import os

EVALUATION_START_DATE_PV = os.environ.get("EVALUATION_START_DATE_PV")
EVALUATION_START_DATE_WEATHER_FORECAST = os.environ.get(
    "EVALUATION_START_DATE_WEATHER_FORECAST"
)

SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")

################## MODEL PARAMETERS ##################
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

################## SEQUENCING PARAMETERS ##################
INPUT_WIDTH = 48
LABEL_WIDTH = 24
SHIFT = 36

##################  VARIABLES  ##################
DATASET = os.environ.get("DATASET")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_JEROME = os.environ.get("GCP_PROJECT_JEROME")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
#
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

##################  CONSTANTS  #####################
absolute_path = os.path.dirname(os.path.abspath(__file__))

LOCAL_DATA_PATH = os.path.join(absolute_path, ".local_data", "mlops", "data")
LOCAL_REGISTRY_PATH = os.path.join(
    absolute_path, ".local_data", "mlops", "training_outputs"
)
