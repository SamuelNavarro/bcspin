# Sproxxo Fraud Detection MLOps Platform



### Brain dump

- I need to talk about stablishing a channel for dependencies sharing and pinning them between MLOPs and DS, since is the biggest source of problems.

- We would need to first empasize the fact that we don't care that much about where the model was trained. Since this is for a real-time inference system, as long as we are able to provide an endpoint an receive post requests, we are fine regardless of the cloud environment. In that sense, I used a plain fastapi endpoint for the example. We are cloud agnostic and we can adapt it for whatever cloud we need.
The consideration would have been different if the serving would be for batch inference, since we would need to source the data, it would be much better to stay where we have the data already, since we would avoid a lot of the headaches that the cloud env solves for us (in this case, gcp). In that case, I would recommend use vertex ai pipelines.

Having said that, if most of the services live in gcp, I would recommend to stay there an maybe deploy the endpoint in a vertex ai endpoint.

Regardless the previous considerations, deploying a model is a matter of taking a pkl file making it available to receive requests, regardless of where you host it, that be the same fastapi endpoint in ec2 or cloud run, a dedicated vertex ai endpoint of sagemaker endpoint, etc. For the sake of the example, I'm creating an endpoint in fastapi.

To ensure reproductibility of the deployed model, I would strongly recommend two things, the versioning of the model and the pin of dependencies for the image creation. That image should also be versioned. Most of the times, I have seen that a lot of trouble comes from mismatches between the libraries used in training vs what is used during training since data scientist are not that aware of the problems the different libraries may incurr. They usually just pip install "library" and proceed. We should encoure them to list the exact version of the libraries.


If you plan to run it locally, please sync it like:
export UV_PROJECT_ENVIRONMENT=.venv-local && uv sync
otherwise, the .venv will mess up the docker env synce we are mounting the volume.
<img width="1592" height="453" alt="Screenshot from 2025-07-28 02-05-10" src="https://github.com/user-attachments/assets/b4936a09-a134-4208-8328-54014ef922f9" />


- We have the layer of pydantic validations but we do have business validations.

- pre-commit with some usefull hooks.
<img width="655" height="247" alt="Screenshot from 2025-07-28 02-00-25" src="https://github.com/user-attachments/assets/c1809006-247f-474f-ada6-392d74c82b30" />


- tox usage
- github actions
- coverage (low threshold for demostration purposes: 40 %)
- cookiecutter for ds projects
-

Branch protection for qa and main ofc.

As integration test we have the sproxxo-api-client

For CD, I really like how jenkins manage deployments when merging to dev, qa and main. That is usually managed by a dedicated devops team, but I really like how we can see and debug the builds there.


## **Transaction Validation & Risk Assessment**

• **Business Rule Validation**: Beyond Pydantic's type checking in `TransactionFeatures`, custom validators enforce business logic (amount limits, valid merchant categories, coordinate ranges, reasonable time values)
• **Automatic Risk Scoring**: Real-time calculation of risk indicators including high amounts (>$500), foreign transactions, late-night activity (midnight-5am), new cards (<30 days), and suspicious transaction patterns
• **Enhanced API Responses**: Fraud detection endpoints now return validation status, specific error details, risk indicator breakdown, and overall risk scores (0-1) alongside ML predictions
• **Intelligent Monitoring**: Automatic logging of invalid transactions and high-risk activities (score >0.7) for improved fraud detection oversight and system monitoring
• **Dual-Layer Protection**: Pydantic ensures data type safety while custom validators catch business rule violations, creating comprehensive input validation for the fraud detection pipeline

🔄 Only Pydantic validation in TransactionFeatures model (basic type checking)
🔄 No business rule validation in the API endpoints
🔄 No risk assessment before fraud prediction


- Talk about locust


### Your Task:

You need to develop a comprehensive plan and outline the steps required to take this model from its current state (a trained file) to a fully deployed and monitored production service. Given Sproxxo nascent MLOps capabilities, your plan should be both effective and mindful of starting from scratch and have the entire freedom to purpose the best solution you identify. Prepare a presentation to expose the plan to the head of the Analytics team in English in the format you consider more convenient. Make the assumptions you need and specify them to work the case.The presentation should present a clear, high-level diagram of your proposed end-to-end MLOps architecture. This should visually represent the flow from data source (BigQuery for training, transaction systems for real-time inference inputs) to model serving, output integration, and monitoring and address at least but not limited the following points:



#### Cloud Provider Recommendation & Justification:
- [ ] Propose a specific cloud provider to the Sproxxo business team. Justify your choice based on factors relevant to a startup with limited existing infrastructure and a small data science team, as well as the existing data location in BigQuery and the nature of Sproxxo's financial products (debit card, digital app, credit products).

Given that, the app is deployed in https://sproxxo-127578390616.northamerica-south1.run.app/. Checkout the docs!!: https://sproxxo-127578390616.northamerica-south1.run.app/docs


#### Model Packaging and Versioning:
- [ ] How would you package this model for deployment?
- [ ] What strategy would you employ for versioning the model and its associated artifacts, especially given that you're establishing these practices?
- [ ] How would you ensure reproducibility of the deployed model?
- [ ]

Right now I have the models locally in an `artifacts` folder. We should store the pkl files in a Cloud Storage.
<img width="1672" height="1000" alt="Screenshot from 2025-07-28 01-54-55" src="https://github.com/user-attachments/assets/ddabf0af-ae50-4a36-affb-7ccefc66bb5c" />
<img width="1672" height="1000" alt="Screenshot from 2025-07-28 01-54-06" src="https://github.com/user-attachments/assets/f41ecb89-04b1-47a8-9178-8fde665f759b" />

<img width="1723" height="612" alt="Screenshot from 2025-07-28 01-52-23" src="https://github.com/user-attachments/assets/183f759f-39ff-486c-ad1d-3d9241d43cd1" />
<img width="641" height="130" alt="Screenshot from 2025-07-27 23-41-34" src="https://github.com/user-attachments/assets/57001063-2637-47aa-8987-88787d36b3c3" />




#### Deployment Strategy:
- [ ] Describe your chosen deployment strategy. Justify your choice based on the real-time fraud detection needs of debit and credit transactions.Outline the minimal yet effective infrastructure components required for this deployment within your chosen cloud environment.

#### CI/CD Pipeline Design:

Good CI/CD design always imply a set of tools with correct usage. I would highly encourage what is set in this repo:
- tox for testing
<img width="570" height="196" alt="Screenshot from 2025-07-28 01-51-37" src="https://github.com/user-attachments/assets/ec3d6b1a-b1a9-4864-ab2b-ea67c59b1549" />


- [ ] Sketch out a CI/CD pipeline for this model. Present a diagram of your proposed CI/CD pipeline detailing each stage.
- [ ] What automated tests would you implement at each stage?
<img width="1182" height="533" alt="Screenshot from 2025-07-27 23-21-45" src="https://github.com/user-attachments/assets/6135d63c-bac7-44aa-8dd9-6b21ad1bd917" />
<img width="557" height="166" alt="Screenshot from 2025-07-27 23-02-02" src="https://github.com/user-attachments/assets/f115e8ec-d4d7-47de-9841-918ee079ce4e" />

<img width="1177" height="702" alt="Screenshot from 2025-07-28 01-49-00" src="https://github.com/user-attachments/assets/0c1a6b03-ec95-49af-9b35-0229960f6e58" />
<img width="727" height="754" alt="Screenshot from 2025-07-28 00-21-28" src="https://github.com/user-attachments/assets/9cae8593-a069-40f2-915e-ba84b248f2af" />
<img width="1255" height="703" alt="Screenshot from 2025-07-28 00-20-13" src="https://github.com/user-attachments/assets/0ae66f3a-fc38-4be1-b15e-4383ac09a8ed" />
<img width="918" height="544" alt="Screenshot from 2025-07-28 00-07-59" src="https://github.com/user-attachments/assets/1cdf0f1b-232b-4f42-aa00-6c9e959ab655" />





#### Monitoring and Alerting:
- [ ] Need to implement the monitoring part
- [ ] What key metrics would you monitor for the deployed model ? Specifically consider metrics relevant to financial transactions and real-time fraud detection.
- [ ] How would you set up alerting for anomalies in these metrics given your chosen cloud provider?
- [ ] What tools and dashboards would you use for visualization and operational insights?

#### Integration Strategy:
- [ ] How would the real-time transaction processing system integrate with your deployed model for inference requests and receiving predictions? What protocols or mechanisms would you use?

























A comprehensive MLOps platform for real-time fraud detection in financial transactions, built with modern Python tooling and best practices.

## 🚀 Features

- **Real-time Fraud Detection**: FastAPI-based API with sub-100ms response times
- **Model Management**: Version control and model lifecycle management
- **Comprehensive Monitoring**: Structured logging and MLflow experiment tracking
- **Production Ready**: Docker containers, CI/CD pipelines, and security best practices
- **Modern Tooling**: uv for dependency management, pre-commit hooks, and comprehensive testing


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
│   │   └── metrics.py          # Metrics collection for GCP
│   ├── training/               # Model training pipeline
│   │   └── train.py            # Training script with MLflow
│   ├── utils/                  # Utility functions
│   │   ├── data_generator.py   # Synthetic data generation
│   │   └── validators.py       # Transaction validation logic
│   └── config.py               # Application configuration
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── performance/            # Performance tests
├── examples/                   # Usage examples
│   ├── api_client.py           # API client example
│   └── sample_input.json       # Sample transaction data
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

## 🛠️ Technology Stack

- **MLflow** for experiment tracking
- **Docker** for containerization
- **GitHub Actions** for CI/CD
- **pre-commit** for code quality

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- uv (Python package manager)



## 📊 API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /health` - Service health check
- `POST /predict` - Single transaction prediction
- `POST /predict/batch` - Batch transaction prediction

### Model Management

- `GET /models` - List all model versions
- `GET /models/{version}` - Get model information
- `POST /models/{version}/activate` - Activate model version
- `DELETE /models/{version}` - Delete model version

## 🔧 Configuration

Environment variables can be set in `.env` file:

```env
ENVIRONMENT=development
LOG_LEVEL=INFO
MODEL_PATH=models/fraud_detection_model.pkl
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Dashboards
- **MLflow**: Experiment tracking and model versioning and registry

## 🧪 Testing Strategy

### Test Types

- **Unit Tests**: Individual component testing
- **Integration Tests**: API and database integration
- **Performance Tests**: Load testing and latency measurement
- **Security Tests**: Vulnerability scanning and penetration testing

TO BE IMPLEMENTED

### Test Coverage
List here the html


## 🚀 Deployment

```
# Access services
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Jupyter: http://localhost:8888
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/SamuelNavarro/bcspin/issues)
