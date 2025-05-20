FROM python:3.9-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY dev-requirements.txt .
RUN pip install --no-cache-dir -r dev-requirements.txt


COPY src/                                 src/
COPY models/                              models/
COPY best_model/                          best_model/ 
COPY serve.py                             serve.py
COPY README.md                            README.md

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
