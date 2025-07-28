# Sproxxo Fraud Detection MLOps Platform



### Mine

- I need to talk about stablishing a channel for dependencies sharing and pinning them between MLOPs and DS, since is the biggest source of problems.

- We would need to first empasize the fact that we don't care that much about where the model was trained. Since this is for a real-time inference system, as long as we are able to provide an endpoint an receive post requests, we are fine regardless of the cloud environment. In that sense, I used a plain fastapi endpoint for the example. We are cloud agnostic and we can adapt it for whatever cloud we need.
The consideration would have been different if the serving would be for batch inference, since we would need to source the data, it would be much better to stay where we have the data already, since we would avoid a lot of the headaches that the cloud env solves for us (in this case, gcp). In that case, I would recommend use vertex ai pipelines.

Having said that, if most of the services live in gcp, I would recommend to stay there an maybe deploy the endpoint in a vertex ai endpoint.

Regardless the previous considerations, deploying a model is a matter of taking a pkl file making it available to receive requests, regardless of where you host it, that be the same fastapi endpoint in ec2 or cloud run, a dedicated vertex ai endpoint of sagemaker endpoint, etc. For the sake of the example, I'm creating an endpoint in fastapi.

To ensure reproductibility of the deployed model, I would strongly recommend two things, the versioning of the model and the pin of dependencies for the image creation. That image should also be versioned. Most of the times, I have seen that a lot of trouble comes from mismatches between the libraries used in training vs what is used during training since data scientist are not that aware of the problems the different libraries may incurr. They usually just pip install "library" and proceed. We should encoure them to list the exact version of the libraries.


If you plan to run it locally, please sync it like:
export UV_PROJECT_ENVIRONMENT=.venv-local && uv sync
otherwise, the .venv will mess up the docker env synce we are mounting the volume.


- We have the layer of pydantic validations but we do have business validations.

- pre-commit with some usefull hooks.
- tox usage
- github actions
- coverage (low threshold for demostration purposes: 40 %)
- cookiecutter for ds projects






Here are concise bullet points for your README:

## **Transaction Validation & Risk Assessment**

• **Business Rule Validation**: Beyond Pydantic's type checking in `TransactionFeatures`, custom validators enforce business logic (amount limits, valid merchant categories, coordinate ranges, reasonable time values)
• **Automatic Risk Scoring**: Real-time calculation of risk indicators including high amounts (>$500), foreign transactions, late-night activity (midnight-5am), new cards (<30 days), and suspicious transaction patterns
• **Enhanced API Responses**: Fraud detection endpoints now return validation status, specific error details, risk indicator breakdown, and overall risk scores (0-1) alongside ML predictions
• **Intelligent Monitoring**: Automatic logging of invalid transactions and high-risk activities (score >0.7) for improved fraud detection oversight and system monitoring
• **Dual-Layer Protection**: Pydantic ensures data type safety while custom validators catch business rule violations, creating comprehensive input validation for the fraud detection pipeline

What's ACTUALLY Being Used:
🔄 Only Pydantic validation in TransactionFeatures model (basic type checking)
🔄 No business rule validation in the API endpoints
🔄 No risk assessment before fraud prediction


- Talk about locust


### Your Task:

You need to develop a comprehensive plan and outline the steps required to take this model from its current state (a trained file) to a fully deployed and monitored production service. Given Sproxxo nascent MLOps capabilities, your plan should be both effective and mindful of starting from scratch and have the entire freedom to purpose the best solution you identify. Prepare a presentation to expose the plan to the head of the Analytics team in English in the format you consider more convenient. Make the assumptions you need and specify them to work the case.The presentation should present a clear, high-level diagram of your proposed end-to-end MLOps architecture. This should visually represent the flow from data source (BigQuery for training, transaction systems for real-time inference inputs) to model serving, output integration, and monitoring and address at least but not limited the following points:








#### Cloud Provider Recommendation & Justification:
- [ ] Propose a specific cloud provider to the Sproxxo business team. Justify your choice based on factors relevant to a startup with limited existing infrastructure and a small data science team, as well as the existing data location in BigQuery and the nature of Sproxxo's financial products (debit card, digital app, credit products).

#### Model Packaging and Versioning:
- [ ] How would you package this model for deployment?
- [ ] What strategy would you employ for versioning the model and its associated artifacts, especially given that you're establishing these practices?
- [ ] How would you ensure reproducibility of the deployed model?

#### Deployment Strategy:
- [ ] Describe your chosen deployment strategy. Justify your choice based on the real-time fraud detection needs of debit and credit transactions.Outline the minimal yet effective infrastructure components required for this deployment within your chosen cloud environment.

#### CI/CD Pipeline Design:

Good CI/CD design always imply a set of tools with correct usage. I would highly encourage what is set in this repo:
- tox for testing


- [ ] Sketch out a CI/CD pipeline for this model. Present a diagram of your proposed CI/CD pipeline detailing each stage.
- [ ] What automated tests would you implement at each stage?







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

## 🏗️ Architecture

### Cloud Provider Recommendation: **Google Cloud Platform (GCP)**

**Justification:**
- **BigQuery Integration**: Existing data in BigQuery for seamless integration
- **Vertex AI**: Native ML model serving with auto-scaling
- **Cloud Run**: Serverless containers for cost-effective scaling
- **Cloud Monitoring**: Ready for GCP Cloud Monitoring integration
- **Cost Efficiency**: Pay-per-use model ideal for startup scaling

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

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sproxxo/sproxxo-mlops.git
   cd sproxxo-mlops
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Setup pre-commit hooks:**
   ```bash
   pre-commit install
   ```

4. **Start the development environment:**
   ```bash
   docker-compose up -d
   ```

### Development


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

## 📈 Monitoring & Observability

### Metrics Collected

- **Request Metrics**: Rate, duration, status codes
- **Prediction Metrics**: Fraud rate, confidence scores, model performance
- **System Metrics**: CPU, memory, active connections
- **Business Metrics**: Transaction volume, fraud detection rate

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
- [Architecture Overview](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Monitoring Guide](docs/monitoring.md)

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use conventional commit messages

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/sproxxo/sproxxo-mlops/issues)
