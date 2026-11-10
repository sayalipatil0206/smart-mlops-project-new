Smart MLOps Project – Breast Cancer Classification

Project Description

This project is an educational Machine Learning and MLOps project. It uses the Breast Cancer Wisconsin dataset to classify tumors into two categories: Malignant and Benign.

The project demonstrates the complete ML workflow, from data preparation and model training to API deployment using Docker.

Technologies Used

- Python 3.11
- Scikit-learn
- Pandas and NumPy
- MLflow
- DVC
- FastAPI
- Pytest
- Docker
- Git and GitHub
- GitHub Actions
- Matplotlib
- SciPy

Project Features

- Dataset loading and preprocessing
- Logistic Regression model
- Grid Search and Randomized Search
- Model evaluation using performance metrics
- MLflow experiment tracking
- DVC data version control
- Model saving using Joblib
- FastAPI prediction API
- Automated testing using Pytest
- Docker-based deployment
- GitHub Actions CI pipeline
- Automated test coverage reporting

Project Structure

smart-mlops-project/
│
├── artifacts/
│   ├── best_model.joblib
│   └── results.json
│
├── data/
├── feature_repo/
├── src/
│   ├── api.py
│   ├── data.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_train.py
│
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── dvc.yaml
└── README.md

Installation and Setup

Clone the repository:

git clone https://github.com/sayalipatil0206/smart-mlops-project-new.git

Move into the project folder:

cd smart-mlops-project-new

Create a virtual environment:

python -m venv .venv

Activate the environment:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Model Training

Run the following command to train the machine learning model:

python -m src.train

The trained model and evaluation results are saved in the "artifacts" folder.

Run the API

Start the FastAPI application:

uvicorn src.api:app --reload

Open the API documentation:

http://127.0.0.1:8000/docs

API Endpoints

- "GET /health" – Checks API and model status.
- "POST /predict" – Predicts the class using 30 numerical features.

Run Tests

Execute the test cases:

python -m pytest -q

Test Results and Code Coverage

The project includes 9 automated test cases covering API, data processing, and model training.

- Total Tests: 9
- Tests Passed: 9
- Tests Failed: 0
- Overall Code Coverage: 98%

Run tests with coverage:

python -m pytest --cov=src --cov-report=term-missing

All 9 test cases passed successfully, with 98% overall code coverage.

Docker Deployment

Build the Docker image:

docker build -t smart-mlops-api .

Run the Docker container:

docker run -d -p 8000:8000 --name smart-mlops-container smart-mlops-api

Open the API documentation:

http://localhost:8000/docs

Check API health:

http://localhost:8000/health

Important Note

This project is developed for educational purposes only. The predictions are based on a sample dataset and must not be used for real medical diagnosis or treatment decisions.