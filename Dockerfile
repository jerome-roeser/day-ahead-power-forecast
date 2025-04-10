FROM python:3.10.6-buster

WORKDIR /prod

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Then only, install power
COPY day_ahead_power_forecast day_ahead_power_forecast

EXPOSE $PORT

CMD ["uvicorn", "day_ahead_power_forecast.api.fast:app", "--host", "0.0.0.0"]
