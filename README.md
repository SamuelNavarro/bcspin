# Sproxxo Fraud Detection MLOps Platform

I've decided to build the platform itself in order to showcase how I would transition from a trained file to a fully deployed and monitored prod service.

The endpoint is live!! try it out: https://sproxxo-127578390616.northamerica-south1.run.app/docs#/

> The swagger ui is really slow due to a really small instance that I used in `GCP` to avoid incurring costs.

or if your prefer, a direct curl would work as well:

```shell
curl -X 'POST' \
  'https://sproxxo-127578390616.northamerica-south1.run.app/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "transaction_id": "string",
  "features": {
    "transaction_amount": 15,
    "merchant_category": "retail",
    "time_of_day": 0,
    "location_lat": 0,
    "location_lon": 0,
    "average_spend": 0,
    "transactions_last_hour": 0,
    "card_age_days": 0,
    "is_foreign_transaction": true,
    "merchant_risk_score": 0
  }
}'
```

You would get something like:

```json
{
    "transaction_id": "string",
    "fraud_probability": 0.2732921242713928,
    "is_fraud": false,
    "confidence_score": 0.5267078757286072,
    "model_version": "20250727_233143",
    "prediction_timestamp": "2025-07-28T10:43:52.166375",
    "feature_importance": {
        "transaction_amount_scaled": 0.0365694984793663,
        "merchant_category_encoded": 0.03143179044127464,
        "time_of_day_scaled": 0.04668349400162697,
        "location_lat_scaled": 0.03449851647019386,
        "location_lon_scaled": 0.032825253903865814,
        "average_spend_scaled": 0.03265126049518585,
        "transactions_last_hour_scaled": 0.0373387448489666,
        "card_age_days_scaled": 0.032754309475421906,
        "is_foreign_transaction": 0.6809464693069458,
        "merchant_risk_score": 0.03430062159895897
    },
    "is_valid": true,
    "validation_errors": {},
    "risk_indicators": {
        "foreign_transaction": 0.8,
        "late_night": 0.6,
        "new_card": 1.0
    },
    "overall_risk_score": 0.8
}
```

## 🚀 Features

### Core ML Capabilities
- **Real-time Fraud Detection**: FastAPI-based API with sub-100ms response times
- **Model Management**: Version control and lifecycle management with MLflow integration
- **Business Rule Validation**: Dual-layer protection with Pydantic + custom business logic validators
- **Risk Assessment**: Automatic calculation of risk indicators (foreign transactions, late-night activity, new cards)
- **Feature Importance**: Real-time explainability with prediction confidence scores

### Production-Ready Infrastructure
- **Containerized Deployment**: Docker containers with pinned dependencies for reproducibility
- **Cloud Agnostic**: Deployable on AWS, GCP, or any container platform
- **Health Monitoring**: Built-in health checks and structured logging with risk indicator tracking
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

### Development & Testing
- **Comprehensive Testing**: Unit, integration, and security tests with 40%+ coverage
- **CI/CD Pipeline**: GitHub Actions with automated testing, linting, and deployment
- **Code Quality**: Pre-commit hooks, tox environments, and automated formatting
- **Security Scanning**: Bandit security analysis and vulnerability detection

### MLOps Best Practices
- **Experiment Tracking**: MLflow integration for model versioning and artifact management
- **Dependency Management**: UV lock files for reproducible environments
- **Shadow Deployment Ready**: Architecture designed for safe production rollouts
- **Model Artifacts**: Versioned pickle files with metadata and performance metrics


## 📁 Project Structure

```
bcspin/
├── sproxxo/                    # Main package
│   ├── api/                    # FastAPI application
│   │   └── main.py             # Main API endpoints
│   ├── models/                 # ML models and inference
│   │   ├── fraud_detector.py   # Core fraud detection model
│   │   └── model_manager.py    # Model versioning and management
│   ├── monitoring/             # Observability and metrics
│   │   ├── logging.py          # Structured logging with structlog
│   ├── training/               # Model training pipeline
│   │   └── train.py            # Training script with MLflow
│   ├── utils/                  # Utility functions
│   │   ├── data_generator.py   # Synthetic data generation
│   │   └── validators.py       # Transaction validation logic
│   └── config.py               # Application configuration
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests. Not implemented just for showcase
│   └── performance/            # Performance tests. Not implemented
├── examples/                   # Usage examples
│   ├── api_client.py           # API client example
│   └── sample_input.json       # Sample transaction data for MLFlow
│   └── request_example.json    # Example of expected payload
├── models/                     # Trained models storage
├── artifacts/                  # Training artifacts
├── mlartifacts/                # MLflow artifacts
├── mlruns/                     # MLflow experiment tracking
├── htmlcov/                    # Test coverage reports
├── docker-compose.yml          # Development environment
├── Dockerfile                  # Container configuration
├── Makefile                    # Development commands
└── pyproject.toml              # Project dependencies and config
```


## 📊 API Endpoints

### Core Endpoints

- `GET /health` - Service health check
- `POST /predict` - Single transaction prediction
- `POST /predict/batch` - Batch transaction prediction

### Model Management

- `GET /models` - List all model versions
- `GET /model/info` - Get model information
- `GET /models/{version}` - Get model version information
- `DELETE /models/{version}` - Delete model version

## Answering the Business Case

We would need to first empasize the fact that we don't care that much about where the model was trained. **Since this is for a real-time inference system**, as long as we are able to provide an endpoint an receive `POST` requests, we are fine regardless of the cloud environment. I used a plain fastapi endpoint for the example. We are cloud agnostic and we can adapt it for whatever cloud env we need.
The consideration would have been different if the serving would be for **batch inference**, since we would need to source the data, it would be easier to stay where we have the data, that would avoid a lot of the headaches that the cloud env solves for us (in this case, gcp).  Having said that, if most of the services live in gcp, I would recommend to stay there an maybe deploy the endpoint in a vertex ai endpoint or Cloud Run.

Regardless the previous considerations, deploying a model is a matter of taking a pkl file and making it available to receive requests, (via fastapi endpoint in ec2 or cloud run, a vertex ai endpoint or sagemaker endpoint, etc). For the sake of the example, I'm creating an endpoint in fastapi.

I have experience in the AWS and since you mention that you use GCP, I decided to deploy it there. The app is deployed in https://sproxxo-127578390616.northamerica-south1.run.app/. You can checkout the swagger ui here: https://sproxxo-127578390616.northamerica-south1.run.app/docs


## Model Packaging and Versioning:

I would highly encourage the usage of some platform for experiment tracking and versioning of the models because that would enable us to pin dependencies and keep track of the versions. For the sake of the example, I used `mlflow` since I remember you mentioned `Databricks` and I read that it has good integration with `MLFlow`. In local development, you can access the UI at `localhost:5000`.

In the image below, you can see the runs that I did to create some simple `artifacts`. How we have version of the model and the dependencies. We are actually pinning the dependency for the Docker container creation. See the [Docker Compose file](https://github.com/SamuelNavarro/bcspin/blob/1caff6419827ae081af39f07960e541bd3895aee/docker-compose.yml#L14-L15)

<img width="1723" height="800" alt="Screenshot from 2025-07-28 01-52-23" src="https://github.com/user-attachments/assets/183f759f-39ff-486c-ad1d-3d9241d43cd1" />

To ensure reproductibility of the deployed model, I would strongly recommend two things:
- The versioning of the model
- Pinning the dependencies for the image creation. That image should also be versioned.
- Usage of docker containers (TODO: Add in docker hub here)

Most of the times, I have seen that a lot of trouble comes from mismatches between the libraries used in training vs what is used during inference since data scientist are not aware of the problems that different libraries may incurr. They usually just pip install "library" and proceed.

We would need to stablish a clear and robust channel between DS and MLOPs, I have seen that having DS provide a full `lock` file works wonders. That's why I'm using `uv` since it locks the dependencies providing a full `uv.lock` from the `pyproject.toml`.


### Proper Documentation of endpoints

FastAPI comes with a builtit Swagger UI

<img width="1713" height="1053" alt="Screenshot from 2025-07-28 11-48-25" src="https://github.com/user-attachments/assets/6e28f839-b5a3-423d-877d-11a60e9604c4" />

## Deployment Strategy:

For the deployment strategy, I would highly recommend `shadow deployment`. What I've seen is that it give us enough time to monitor, test and debug the service from two perspectives:
- MLOPs components: latency, cpu usage, memory utilization, etc.
- DS components: proper data distribution, correct usage of features, missing values, etc.

The `shadow deployment` would requiere some efforts of implementing the stack in GCP or whatever cloud we decide to stay and if there are enough resources, some coordinatoin between the `backend` team since a lot of times some implementations are way easier to do in other components of the `backend` and just keep invoking the model endpoints.

## CI/CD Pipeline Design:

I would highly encourage, among other things, what I used in this repo:

### Test

- **Unit Tests**: Individual component testing. Under [`test`](https://github.com/SamuelNavarro/bcspin/tree/main/tests)
- **Integration Tests**: API, e2e tests. Once such example is in the **Run API client integration test** where we actually build the docker container and invoke some of the endpoints multiple times via the [api_client.py](https://github.com/SamuelNavarro/bcspin/blob/main/examples/api_client.py). You can see the jobs [here](https://github.com/SamuelNavarro/bcspin/actions/runs/16575652923/job/46879265284)

<img width="1177" height="702" alt="Screenshot from 2025-07-28 01-49-00" src="https://github.com/user-attachments/assets/0c1a6b03-ec95-49af-9b35-0229960f6e58" />
<img width="727" height="754" alt="Screenshot from 2025-07-28 00-21-28" src="https://github.com/user-attachments/assets/9cae8593-a069-40f2-915e-ba84b248f2af" />

- **Performance Tests**: Load testing and latency measurement, we could use `Locust`. Not implemented in the repo.
- **Security Tests**: Vulnerability scanning and penetration testing. These are implemented, see in [pyproject security section](https://github.com/SamuelNavarro/bcspin/blob/2d14ee846462df421ea06168f70a96a1c308aa19/pyproject.toml#L131-L134)

#### Test Coverage

For now, we have a really low threshold for the demo (40 %) but usually this value is supposed to stay high.

<img width="802" height="611" alt="Screenshot from 2025-07-28 11-12-55" src="https://github.com/user-attachments/assets/bbce1f77-e16b-4467-b184-76e2cd8d5ba6" />

The coverage allow us to see precisely which parts of the code are not being tested.

<img width="1138" height="946" alt="Screenshot from 2025-07-28 11-15-07" src="https://github.com/user-attachments/assets/41d9f4b7-4e1c-4c55-917e-430fa686ec8d" />



### Tox usage

The envs are declared in the `pyproject.toml` file and you can see that everytime the developer can run formatters, linters, testing, etc. By doing this, we also ensure that the same proceedure is run in Github Actions (see below)

<img width="1182" height="533" alt="Screenshot from 2025-07-27 23-21-45" src="https://github.com/user-attachments/assets/6135d63c-bac7-44aa-8dd9-6b21ad1bd917" />


<img width="557" height="166" alt="Screenshot from 2025-07-27 23-02-02" src="https://github.com/user-attachments/assets/f115e8ec-d4d7-47de-9841-918ee079ce4e" />

<img width="570" height="196" alt="Screenshot from 2025-07-28 01-51-37" src="https://github.com/user-attachments/assets/ec3d6b1a-b1a9-4864-ab2b-ea67c59b1549" />

### Github Actions

I'm intentionally showcasing how we are pass from failing jobs to successful ones since that is usally the flow. We expect to keep failing due to restrict best practices being implemented directly in the repo.

<img width="918" height="544" alt="Screenshot from 2025-07-28 00-07-59" src="https://github.com/user-attachments/assets/1cdf0f1b-232b-4f42-aa00-6c9e959ab655" />

You can see that the same `tox` testing is being executed along some other `jobs` that we implemented. Also, you cna directly download the [`coverate` report from here](https://github.com/SamuelNavarro/bcspin/actions/runs/16575652923) in the Artifacts section.


<img width="1882" height="911" alt="Screenshot from 2025-07-28 11-12-26" src="https://github.com/user-attachments/assets/5b89cc59-d468-4a45-bca0-64eda5b91f5f" />

We get the benfit of alerts if some builds were not successful.

<img width="1592" height="453" alt="Screenshot from 2025-07-28 02-05-10" src="https://github.com/user-attachments/assets/b4936a09-a134-4208-8328-54014ef922f9" />


### For CD
- I really like how jenkins manage deployments when merging to `dev`, `qa` and `main`. That is usually managed by a dedicated devops team, but I really like how we can see and debug the builds there.
- Encourage proper branch protection for qa and main, since merging would trigger deployments.


### Pre-commit with some usefull hooks

<img width="655" height="247" alt="Screenshot from 2025-07-28 02-00-25" src="https://github.com/user-attachments/assets/c1809006-247f-474f-ada6-392d74c82b30" />



## Monitoring and Alerting:

The monitoring and alerting, we would also need to split them into two:
- **MLOPs monitoring**: CPU utilization, memory utilization, etc and we would need alerts in place everytime we hit a predefined threshold.
- **DS monitoring**: Missing rate of features, data distribution, etc. We usually should either receive an email with all these statistics or set up pipelines that trigger alerts as well when we reach some undesired level.


I usually like the usage of `graphana` for the MLOPs monitoring since we can easily set up and customize a dashboard, good slack or email integration, etc.

For data related monitoring, it's good practice to always dump the data in buckets or cloud storage, so we can set up pipelines that read from them and source them into our dashboards. That's why the payload in our response has the `features_importance` key.

A good thing of deploying things in the cloud is that we get to see metrics for the instance right away:

<img width="1711" height="770" alt="Screenshot from 2025-07-28 04-54-36" src="https://github.com/user-attachments/assets/ddf8f3d4-abe5-41e1-acd1-addadcf58644" />


<img width="1711" height="770" alt="Screenshot from 2025-07-28 04-54-46" src="https://github.com/user-attachments/assets/eeacd98a-8a42-4d5b-8c8d-23e3d783f6b6" />

## Integration Strategy:

We have the layer of pydantic validations but we do have business validations. One thing is to validate schemas or some system expectations, but we can stablish some business logic as validation. As you can see, the code has `Transaction Validation` and `Risk Assessment` based on the actual request.


• **Business Rule Validation**: Beyond Pydantic's type checking in `TransactionFeatures`, custom validators enforce business logic (amount limits, valid merchant categories, coordinate ranges, reasonable time values)
• **Automatic Risk Scoring**: Real-time calculation of risk indicators including high amounts, foreign transactions, late-night activity, new cards, and suspicious transaction patterns
• **Enhanced API Responses**: Fraud detection endpoints return validation status, specific error details, risk indicator breakdown, and overall risk scores (0-1) alongside ML predictions
• **Intelligent Monitoring**: Automatic logging of invalid transactions and high-risk activities for improved fraud detection oversight and system monitoring
• **Dual-Layer Protection**: Pydantic ensures data type safety while custom validators catch business rule violations, creating comprehensive input validation for the fraud detection pipeline



## 🛠️ Local Development

### Quick Start

1. **Install dependencies**:
   ```bash
   make install-dev  # Install with development dependencies
   # or
   make install-all  # Install all optional dependencies (dev + mlops)
   ```

2. **Train a model**:
   ```bash
   make train        # Train with 10,000 samples
   # or
   make train-quick  # Quick training with 1,000 samples
   ```

3. **Run with Docker**:
   ```bash
   make docker-sproxxo-api  # Build and run API in Docker
   ```

4. **Run the API locally**:
   ```bash
   make run-api-examples      # Run code under `examples/api_client.py`
   make run-api               # Run Fast API
   ```

### Development Workflow

**Code Quality & Testing**:
```bash
make tox           # Run all tests and checks
make tox-test      # Run tests only
make tox-lint      # Run linting only
make tox-format    # Fix code formatting
make check-all     # Run lint + test + security checks
```

**Useful Commands**:
```bash
make help          # Show all available commands
make list-models   # List available trained models
make check-health  # Check if API is running
make clean         # Clean build artifacts
```

**Docker Development**:
```bash
make docker-compose-up    # Start all services
make docker-compose-down  # Stop all services
make logs                 # View application logs
```


## 📚 Documentation

- [API Documentation](https://sproxxo-127578390616.northamerica-south1.run.app/docs#/)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/SamuelNavarro/bcspin/issues)
