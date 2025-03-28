pickleball-ai-mlops/
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # Continuous Integration
│       ├── cd.yml               # Continuous Deployment
│       └── model_training.yml   # ML Model Training Pipeline
│
├── infrastructure/               # Infrastructure as Code
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── kubernetes/
│       ├── deployment.yml
│       └── service.yml
│
├── src/
│   ├── data_management/
│   │   ├── data_collector.py
│   │   ├── data_validator.py
│   │   └── feature_store.py
│   │
│   ├── preprocessing/
│   │   ├── video_preprocessor.py
│   │   ├── augmentation.py
│   │   └── feature_extractor.py
│   │
│   ├── model/
│   │   ├── vision_model.py
│   │   ├── llm_model.py
│   │   └── model_registry.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── experiment_tracking.py
│   │   └── model_versioning.py
│   │
│   ├── inference/
│   │   ├── batch_predictor.py
│   │   ├── real_time_predictor.py
│   │   └── model_monitor.py
│   │
│   └── utils/
│       ├── config_manager.py
│       ├── logging.py
│       └── metrics.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── notebooks/
│   ├── exploration/
│   └── experiments/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── artifacts/
│
├── mlflow/
│   └── mlflow.db
│
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── inference_config.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── deploy.py
│
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md