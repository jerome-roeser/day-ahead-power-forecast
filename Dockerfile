FROM python:3.10.6-buster

WORKDIR /prod

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Then only, install power
COPY power power
# COPY le-wagon-data-411310-7c498969c3b9.json le-wagon-data-411310-7c498969c3b9.json
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile
# RUN make reset_local_files
# COPY 20240310-115742.h5 /root/.lewagon/mlops/training_outputs/models/20240310-115742.h5

CMD uvicorn power.api.fast:app --host 0.0.0.0 --port $PORT
