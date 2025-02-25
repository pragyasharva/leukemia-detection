# leukemia-detection


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/pragyasharva/leukemia-detection```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n env python=3.8 -y
```

```bash
conda activate env
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui


### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=URI
MLFLOW_TRACKING_USERNAME=USERNAME
MLFLOW_TRACKING_PASSWORD=TOKEN
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/pragyasharva/leukemia-detection.mlflow

export MLFLOW_TRACKING_USERNAME=pragyasharva

export MLFLOW_TRACKING_PASSWORD=7072512c2294bd7b378427f2ba8b8f393315356b

```