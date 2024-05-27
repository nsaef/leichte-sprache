# notes for self

## Set up MLFLow

Start the MLFlow container
```shell
docker run -d -v /home/nasrin/data/mlflow/mlruns:/mlruns -v /home/nasrin/data/mlflow/mlartifacts:/mlartifacts --restart unless-stopped --name mlflow -p 0.0.0.0:5555:8080/tcp ghcr.io/mlflow/mlflow:v2.11.1 mlflow server --host 0.0.0.0 --port 8080
```
MLFlow now runs *in WSL*. To connect to the GUI from Windows, run `ip a` to find out the IP listed under `eth0`. 

- configure .env with MLFLow variables:

```
MLFLOW_EXPERIMENT_NAME = "leichte_sprache"
MLFLOW_TRACKING_URI = "http://172.17.70.208:5555/"
```

- Remember to set the name of the run in the training config (#todo: dynamic run names)
