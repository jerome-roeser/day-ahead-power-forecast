import base64
from email.message import EmailMessage
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from prefect import flow, task

from day_ahead_power_forecast.interface.main import evaluate, preprocess, train
from day_ahead_power_forecast.ml_ops.registry import mlflow_transition_model
from day_ahead_power_forecast.params import (
    GOOGLE_OAUTH_CREDENTIALS,
    PREFECT_FLOW_NAME,
    SENDER_EMAIL,
)


@task
def preprocess_new_data():
    return preprocess()


@task
def evaluate_production_model():
    return evaluate()


@task
def re_train():
    return train()


@task
def transition_model(current_stage: str, new_stage: str):
    return mlflow_transition_model(current_stage=current_stage, new_stage=new_stage)


@task
def notify_via_email(
    old_mae: float, new_mae: float, recipient_email: str
) -> EmailMessage:
    """
    Notify about the performance via e-mail

    Create and send an email message
    Print the returned  message id

    Parameters
    ----------
    old_mae: float
        The old MAE
    new_mae: float
        The new MAE
    recipient_email: str
        The recipient email address

    Returns
    -------
    Message object, including message id

    Notes
    -----
    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """

    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if Path("token.json").exists():
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_OAUTH_CREDENTIALS, SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    if new_mae < old_mae and new_mae < 2.5:
        subject = "🚀 New Model Deployed to Production"
        body = f"New model replacing old in production with MAE: {new_mae}. The old MAE was: {old_mae}."
    elif old_mae < 2.5:
        subject = "✅ Old Model Still Good Enough"
        body = f"Old model still good enough: Old MAE: {old_mae}, New MAE: {new_mae}."
    else:
        subject = "🚨 No Model Good Enough"
        body = f"No model good enough: Old MAE: {old_mae}, New MAE: {new_mae}."

    # Send the email
    try:
        # Call the Gmail API
        service = build("gmail", "v1", credentials=creds)
        message = EmailMessage()

        # headers
        message["To"] = recipient_email
        message["From"] = SENDER_EMAIL
        message["Subject"] = subject

        # text
        message.set_content(body)

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}
        # pylint: disable=E1101
        send_message = (
            service.users().messages().send(userId="me", body=create_message).execute()
        )

        print(f"Message Id: {send_message['id']}")
        return send_message

    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f"An error occurred: {error}")


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Build the Prefect workflow for the app. It should:
        - compute `old_mae` by evaluating the current production model in this new month period
        - compute `new_mae` by re-training, then evaluating the current production model on this new month period
        - if the new one is better than the old one, replace the current production model with the new one
        - send an e-mail notification with the results
    """

    preprocessed = preprocess_new_data.submit()

    old_mae = evaluate_production_model.submit(wait_for=[preprocessed])
    new_mae = re_train.submit(wait_for=[preprocessed])

    old_mae = old_mae.result()
    new_mae = new_mae.result()

    if new_mae < old_mae:
        print(
            f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}"
        )
        transition_model.submit(current_stage="staging", new_stage="production")
    else:
        print(
            f"🚀 Old model kept in place with MAE: {old_mae}. The new MAE was: {new_mae}"
        )

    notify_via_email.submit(old_mae, new_mae, SENDER_EMAIL)


if __name__ == "__main__":
    train_flow()
