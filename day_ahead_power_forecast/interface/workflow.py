import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
from dateutil.relativedelta import relativedelta
from prefect import flow, task

from day_ahead_power_forecast.interface.main import evaluate, preprocess, train
from day_ahead_power_forecast.ml_ops.registry import mlflow_transition_model
from day_ahead_power_forecast.params import (
    EVALUATION_START_DATE_PV,
    PREFECT_FLOW_NAME,
    SENDER_EMAIL,
    SENDER_PASSWORD,
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
def notify(old_mae, new_mae):
    """
    Notify about the performance
    """
    base_url = "https://wagon-chat.herokuapp.com"
    channel = "802"
    url = f"{base_url}/{channel}/messages"
    author = "krokrob"

    if new_mae < old_mae and new_mae < 2.5:
        content = f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}"
    elif old_mae < 2.5:
        content = (
            f"✅ Old model still good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
        )
    else:
        content = f"🚨 No model good enough: Old MAE: {old_mae} - New MAE: {new_mae}"

    data = dict(author=author, content=content)

    response = requests.post(url, data=data)
    response.raise_for_status()


@task
def notify_via_email(old_mae, new_mae, recipient_email):
    """
    Notify about the performance via e-mail
    """
    sender_email = SENDER_EMAIL  # Replace with your email
    sender_password = SENDER_PASSWORD  # Replace with your email password
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server
    smtp_port = 587  # Replace with your SMTP port (e.g., 587 for TLS)

    if new_mae < old_mae and new_mae < 2.5:
        subject = "🚀 New Model Deployed to Production"
        body = f"New model replacing old in production with MAE: {new_mae}. The old MAE was: {old_mae}."
    elif old_mae < 2.5:
        subject = "✅ Old Model Still Good Enough"
        body = f"Old model still good enough: Old MAE: {old_mae}, New MAE: {new_mae}."
    else:
        subject = "🚨 No Model Good Enough"
        body = f"No model good enough: Old MAE: {old_mae}, New MAE: {new_mae}."

    # Create the email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email notification: {e}")


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Build the Prefect workflow for the `taxifare` package. It should:
        - preprocess 1 month of new data, starting from EVALUATION_START_DATE
        - compute `old_mae` by evaluating the current production model in this new month period
        - compute `new_mae` by re-training, then evaluating the current production model on this new month period
        - if the new one is better than the old one, replace the current production model with the new one
        - if neither model is good enough, send a notification!
    """

    min_date = EVALUATION_START_DATE_PV
    max_date = str(
        datetime.strptime(min_date, "%Y-%m-%d") + relativedelta(months=1)
    ).split()[0]

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
