FROM python:3.10.6-buster


WORKDIR /prod


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY day_ahead_power_forecast day_ahead_power_forecast


EXPOSE $PORT
CMD ["uvicorn", "day_ahead_power_forecast.api.fast:app", "--host", "0.0.0.0"]
