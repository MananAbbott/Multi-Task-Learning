import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import os

class EmbeddingModel:
    def __init__(
            self, 
            model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
            device: Optional[str] = None, 
            projection_dim: Optional[int] = None,
            normalize: bool = False
            ):
        
        # Set device based on processor availability
        if not device:
            device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

        # Set the projection dimension to reduce the embedding size
        # Note : Has to be trained for tasks
        self.projection_dim = projection_dim
        if self.projection_dim:
            hidden_size = self.model.config.hidden_size
            self.projection = torch.nn.Linear(hidden_size, self.projection_dim)
            self.projection.to(self.device)
        
        # Set normaliazation
        self.normalize = normalize


    # Encode sentences into embeddings   
    def encode(self, sentences: List[str]):
        inputs = self.tokenizer(
            sentences, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
            
        # Get the CLS token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]


        # Projection
        if self.projection_dim:
            cls_embeddings = self.projection(cls_embeddings)
        
        # L2 Normalization
        if self.normalize:
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
        
        return cls_embeddings
    
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
        if self.projection_dim:
            torch.save(self.projection.state_dict(), f"{path}/projection.pth")
        print(f"Model saved to {path}")
        return self

    def load(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        self.model.to(self.device)
        if self.projection_dim:
            self.projection = torch.nn.Linear(self.model.config.hidden_size, self.projection_dim)
            self.projection.to(self.device)
            self.projection.load_state_dict(torch.load(f"{path}/projection.pth", map_location=self.device))
        print(f"Model loaded from {path}")
        return self