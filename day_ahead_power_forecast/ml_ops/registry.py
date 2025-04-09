import glob
import os
import pickle
import time

import mlflow
import torch
from colorama import Fore, Style
from google.cloud import storage
from mlflow import MlflowClient, MlflowException
from torchinfo import summary

from day_ahead_power_forecast.params import (
    BUCKET_NAME,
    DATASET,
    LOCAL_REGISTRY_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_TARGET,
)


def save_results(
    params: dict,
    metrics: dict,
    history: dict = None,
) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """

    if MODEL_TARGET == "mlflow":
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)

        print("✅ Results saved on MLflow")

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


def save_model(
    model: torch.nn.Module = None, signature: mlflow.models.ModelSignature = None
) -> None:
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

    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", DATASET, f"{timestamp}.pt")
    torch.save(model, model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nSave model to GCS @ {BUCKET_NAME}..." + Style.RESET_ALL)

        model_filename = model_path.split("/")[-1]  # e.g. "20250326-161047.pt"
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{DATASET}/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    elif MODEL_TARGET == "mlflow":
        print(Fore.BLUE + "\nSave model to mlflow ..." + Style.RESET_ALL)

        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        summary_path = os.path.join(
            LOCAL_REGISTRY_PATH, "summaries", timestamp + ".txt"
        )

        with open(summary_path, "w") as f:
            f.write(str(summary(model, verbose=2)))
        mlflow.log_artifact(summary_path)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            signature=signature,
            artifact_path="model",
            registered_model_name=f"dev.ml_team.{DATASET}.{MLFLOW_MODEL_NAME}",
        )

        print("✅ Model saved to MLflow")

        return None

    return None


def load_model(stage="production") -> torch.nn.Module:
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
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", DATASET)
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

            print(f"✅ Latest model downloaded from cloud storage ({latest_blob.name})")

            return latest_model
        except Exception as e:
            print(e)
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    elif MODEL_TARGET == "mlflow":
        print(Fore.BLUE + "\nLoad latest model from mlflow ..." + Style.RESET_ALL)

        environments = ["dev", "staging", "production", "archived"]
        try:
            assert stage in environments
        except AssertionError:
            print(f"Please choose valid model environments from {environments}")
            return None

        client = MlflowClient()
        registered_model_name = f"{stage}.ml_team.{DATASET}.{MLFLOW_MODEL_NAME}"

        try:
            latest_registered_model = client.get_registered_model(
                registered_model_name
            ).latest_versions[0]
        except MlflowException:
            print(
                f"\n❌ No model found with name {registered_model_name} \
                    in stage {stage}"
            )
            return None

        logged_model = (
            f"models:/{registered_model_name}/{latest_registered_model.version}"
        )
        latest_model = mlflow.pytorch.load_model(logged_model)

        print(f"✅ Latest model downloaded from mlflow ({logged_model})")
        return latest_model

    else:
        return None


def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """

    environments = ["dev", "staging", "production", "archived"]
    try:
        assert current_stage and new_stage in environments
    except AssertionError:
        print(f"Please choose valid model environments from {environments}")
        return None

    client = MlflowClient()
    src_name = f"{current_stage}.ml_team.{MLFLOW_MODEL_NAME}"

    # Copy the source model version into a new registered model
    mv_src = client.get_registered_model(src_name).latest_versions[0]
    dst_name = f"{new_stage}.ml_team.{DATASET}.{MLFLOW_MODEL_NAME}"
    src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
    client.copy_model_version(src_model_uri, dst_name)

    print(
        f"✅ Model {MLFLOW_MODEL_NAME} (version {mv_src.version}) \
            transitioned from {current_stage} to {new_stage}"
    )

    return None


##### Wrapper with autolog works only with pytorch Lightning ####################
def mlflow_run(func, params: dict = None, context: str = None):
    """
    Generic function to log params and results to MLflow along with
    Pytorch auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow.
          Defaults to None.
        - context (str, optional): Param describing the context of the run.
          Defaults to "Train".
    """

    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run() as run:
            mlflow.pytorch.autolog()
            results = func(*args, **kwargs)
            print(run.info)

        print("✅ mlflow_run auto-log done")

        return results

    return wrapper
