Sample project structure for a machine learning project
```text
my_project/
│
├── configs/
│   ├── base.yaml
│   ├── train.yaml
│   ├── eval.yaml
│   └── deploy.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── make_dataset.py      # script to download/process raw → processed
│
├── src/
│   ├── my_project/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── dataloader.py
│   │   │   ├── transforms.py
│   │   │   └── dataset.py
│   │   ├── models/
│   │   │   ├── model_arch.py
│   │   │   ├── loss.py
│   │   │   └── metrics.py
│   │   ├── training/
│   │   │   ├── trainer.py
│   │   │   ├── callbacks.py
│   │   │   └── lr_scheduler.py
│   │   ├── evaluation/
│   │   │   ├── evaluate.py
│   │   │   └── visualize.py
│   │   ├── inference/
│   │   │   ├── predict.py
│   │   │   └── serveModel.py
│   │   └── utils/
│   │       ├── logging.py
│   │       ├── config_helpers.py
│   │       └── io_helpers.py
│
├── notebooks/
│   ├── eda/
│   ├── experiments/
│   └── results_visualization.ipynb
│
├── scripts/
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── run_inference.py
│
├── experiments/
│   ├── exp_001/
│   │   ├── checkpoints/
│   │   ├── metrics.json
│   │   └── config_used.yaml
│   └── exp_002/...
│
├── models/
│   └── final_model.pth / .h5 etc
│
├── logs/
│   └── tensorboard/
│
├── deploy/
│   ├── Dockerfile
│   ├── api_server.py
│   └── requirements.txt   # for deployment environment possibly separate
│
├── tests/
│   ├── test_dataloader.py
│   ├── test_model_architecture.py
│   └── test_training_loop.py
│
├── docs/
│   ├── architecture.md
│   └── usage.md
│
├── .gitignore
├── README.md
├── LICENSE
├── requirements.txt or pyproject.toml
└── setup.py (if packaging as a library)
```