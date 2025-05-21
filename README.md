## Task Associated files

  All explanation related to the assesment are present in **Fetch_assesment.pdf** file.

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
### 2. (Optional) Docker setup

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

Note: All task related explanations are present in the Fetch_assesmet.pdf file