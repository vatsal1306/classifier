Sample project structure for a machine learning project
```text
classifier/
├── .gitignore
├── .python-version
├── Makefile
├── README.md
├── docker
    ├── Dockerfile
    └── docker-compose.yaml
├── download_from_gdrive.py
├── misc.py
├── requirements.txt
├── src
    ├── __init__.py
    ├── config.py
    ├── data
    │   ├── __init__.py
    │   ├── dataloader.py
    │   ├── make_dataset.py
    │   └── transformations.py
    ├── evaluate.py
    ├── inference.py
    ├── models.py
    ├── predict.py
    ├── train.py
    ├── utils
    │   ├── logger.py
    │   ├── utils.py
    │   └── viz.py
    └── viz_predictions.py
├── test.sh
├── train.sh
├── upload.sh
└── upload_to_drive.py
```