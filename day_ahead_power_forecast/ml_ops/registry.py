import glob
import os
import pickle
import time

import torch
from colorama import Fore, Style
from google.cloud import storage

from day_ahead_power_forecast.params import (
    BUCKET_NAME,
    DATASET,
    LOCAL_REGISTRY_PATH,
    MODEL_TARGET,
)


def save_results(params: dict, metrics: dict, history: dict = None) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(
            LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle"
        )
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # Save history locally
    if history is not None:
        history_path = os.path.join(
            LOCAL_REGISTRY_PATH, "histories", timestamp + ".pickle"
        )
        with open(history_path, "wb") as file:
            pickle.dump(history, file)

    print("✅ Results saved locally")


def save_model(model: torch.nn.Module = None) -> None:
    """
    Persist trained model locally on the hard drive at
                f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.pt"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at
                "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS
                (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    if DATASET == "forecast":
        model_path = os.path.join(
            LOCAL_REGISTRY_PATH, "models", "full", f"{timestamp}.pt"
        )
    else:
        model_path = os.path.join(
            LOCAL_REGISTRY_PATH, "models", "pv", f"{timestamp}.pt"
        )
    torch.save(model, model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nSave model to GCS @ {BUCKET_NAME}..." + Style.RESET_ALL)

        model_filename = model_path.split("/")[
            -1
        ]  # e.g. "20250326-161047.pt" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        if DATASET == "forecast":
            blob = bucket.blob(f"models/full/{model_filename}")
        else:
            blob = bucket.blob(f"models/pv/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None


def load_model(stage="Production") -> torch.nn.Module:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(
            Fore.BLUE + "\nLoad latest model from local registry..." + Style.RESET_ALL
        )

        # Get the latest model version name by the timestamp on disk
        if DATASET == "forecast":
            local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", "full")
        else:
            local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", "pv")

        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + "\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = torch.load(most_recent_model_path_on_disk, weights_only=False)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + "\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(
                LOCAL_REGISTRY_PATH, latest_blob.name
            )
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = torch.load(latest_model_path_to_save, weights_only=False)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except Exception as e:
            print(e)
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    else:
        return None
