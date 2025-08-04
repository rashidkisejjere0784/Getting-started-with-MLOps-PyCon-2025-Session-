# Getting Started with MLOps - PyCon Uganda 2025

## What is MLOps?

MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently. It encompasses the entire ML lifecycle including data preparation, model training, deployment, monitoring, and maintenance.

Key benefits of MLOps include:
- Reproducible ML workflows
- Automated model deployment
- Continuous integration and delivery for ML
- Model monitoring and governance
- Scalable infrastructure management

## About This Repository

This repository demonstrates MLOps concepts and practices using **ZenML**, a modern MLOps framework that creates reproducible ML pipelines. ZenML helps you build portable, production-ready ML workflows with minimal overhead.

## Prerequisites

### Installing uv

First, install `uv`, a fast Python package installer and resolver:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/rashidkisejjere0784/Getting-started-with-MLOps-PyCon-2025-Session-
    cd Getting-started-with-MLOps-PyCon-2025-Session
    ```

2. **Initialize the project environment:**
    ```bash
    uv init
    ```

3. **Install dependencies:**
    ```bash
    uv pip install -r pyproject.toml
    ```

## Running the Project

Execute the main MLOps pipeline:

```bash
uv run main.py
```

## Next Steps

- Check out ZenML documentation: https://docs.zenml.io/
- Experiment with different ML models and data sources