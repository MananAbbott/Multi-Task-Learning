## Project Overview

This repository contains my end-to-end solution for the **Fetch Rewards – ML Engineer Apprentice** take-home.
The objective is to transform raw sentences into vector representations **and** predict their **topic** (4 classes) and **sentiment** (3 classes) in a single, lightweight pipeline.

| Task                        | Deliverable                                                                                                  | Location                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| **1 – Sentence embeddings** | `EmbeddingModel` — MiniLM backbone with CLS pooling, optional projection, and L2-norm                        | `src/embedding_model.py` · `task1.ipynb` |
| **2 – Multi-task head**     | Shared encoder + twin classification heads (topic & sentiment)                                               | `src/mtl.py`                             |
| **3 – Training strategies** | Notebook containing **three-phase** (freeze → unfreeze) fine-tuning strategy; includes an alternative training strategy too | `task2_3_4.ipynb`                        |
| **4 – Production API**      | FastAPI service exposing `/predict` routes                                                      | `serve.py`                               |
| **Bonus**                   | Fully containerised workflow that builds the image and fires sample requests                                 | `Dockerfile` · `build_and_run.sh`        |


## Task Associated files

  All explanation related to the assesment are present in [Fetch_assessment.pdf](./Fetch_assessment.pdf) file.

  Files associated to Task 1 are **src/embedding_model.py** and **task1.ipynb**.

  Files associated to Task 2 are **src/mtl.py** and **task2_3_4.ipynb**.

  Files associated to Task 3 are **src/mtl.py** and **task2_3_4.ipynb**.

  Files associated to Task 4 are **src/mtl.py** and **task2_3_4.ipynb**.

## Codebase Overview

Below is a mapping of every file and directory in this repo to its purpose:

- **best_model/**  
  Contains the final selected model checkpoint used for inference.

- **data/synthetic_data.csv**  
  Synthetic dataset for demonstrating multi-task training.

- **dockerfile**  
  Docker recipe to build a container with all dependencies.

- **dev-requirements.txt**  
  Development‐only Python dependencies.

- **build_and_run.sh**  
  Helper script: builds the Docker image and runs it and tests on two examples.

- **models/**  
  Directory where backbone trained model checkpoints are saved.

- **requirements.txt**  
  Core Python dependencies for running the notebooks and scripts.

- **README.md**  
  This documentation file—setup, usage, and codebase overview.

- **serve.py**  
  FastAPI app exposing endpoints for embedding and classification.

- **src/embedding_model.py**  
  Implements **Task 1** – sentence embedding module with CLS pooling, optional projection & L2‐norm.

- **src/mtl.py**  
  Implements **Tasks 2 & 3** – `MultiTaskModel` with shared backbone + two heads (4-way topic & sentiment classification).

- **task1.ipynb**  
  Interactive notebook demo for **Task 1**: loading `EmbeddingModel`, encoding sentences, inspecting outputs.

- **task2_3_4.ipynb**  
  Interactive notebook combining **Tasks 2, 3 & 4**: building the multi-task model, two training strategies (first phase training and an alternate to train all parameters directly), evaluation, and inference.

- **Fetch_assesment.pdf**  
  PDF file containing explanation related to choices made for all the four tasks and more brief explanations.


## Repository Structure
    ├── best_model/               # Final selected model checkpoint for inference
    ├── data/
    │ └── synthetic_data.csv      # Synthetic dataset for multi-task demos
    ├── models/                   # Directory for saving trained model checkpoints
    ├── src/
    │ ├── embedding_model.py      # Task 1: Sentence embedding implementation
    │ └── mtl.py                  # Tasks 2 & 3: MultiTaskModel definition & utilities
    ├── task1.ipynb               # Interactive notebook for Task 1 demo
    ├── task2_3_4.ipynb           # Interactive notebook for Tasks 2, 3 & 4 (training, eval, inference)
    ├── serve.py                  # FastAPI app exposing embedding & classification endpoints
    ├── build_and_run.sh          # Script to build + run the Docker container with test examples
    ├── dockerfile                # Dockerfile for containerized environment
    ├── dev-requirements.txt      # Development-only Python deps (FastAPI, Uvicorn, Jupyter, etc.)
    ├── requirements.txt          # Core Python dependencies
    ├── README.md                 # This documentation file
    ├── Fetch_assesment.pdf       # Explanation related to all the fours tasks
    ├── .dockerignore             # Files/folders ignored by Docker build
    └── .gitignore                # Files/folders ignored by Git


## Installation

### Prerequisites
- **Git** (to clone the repo)  
- **Python 3.9+**  
- **pip** (Python package installer)  
- **Docker** & **docker-compose** (optional, for containerized setup)

---

### 1. Clone the repository

```bash
git clone https://github.com/your-org/fetch-assessment.git
cd fetch-assessment
```
### 2. (Quick Demo) Docker setup

Build the Docker image and run the default demo/training pipeline:

```bash
# Build the image and run the container with a couple examples
./build_and_run.sh
```

### 3. Local virtual environment setup (For notebooks)

1. **Create & activate a virtual environment**  

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Upgrade pip**  
   ```bash
   pip install --upgrade pip
   ```

3. **Install core dependencies**  
   ```bash
   pip install -r requirements.txt
    ```
4. **Run the .ipynb notebooks for interactive demos**  

Note: All task related explanations are present in the Fetch_assesment.pdf file


## Results & Metrics

| Metric     | A (Baseline) | A (Post-train) | B (Baseline) | B (Post-train) |
|------------|--------------|----------------|--------------|----------------|
| Accuracy   | 0.20         | 0.80           | 0.30         | 0.6333         |
| Recall     | 0.00         | 1.00           | 0.00         | 0.60           |
| Precision  | 0.00         | 1.00           | 0.00         | 0.8571         |

*A* and *B* correspond to the two prediction heads (i.e., topic and sentiment).  
 Full training logs and metric-calculation code present in `task2_3_4.ipynb`.


