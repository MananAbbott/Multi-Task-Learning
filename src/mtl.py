from src.embedding_model import EmbeddingModel
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from typing import Optional, Tuple


class MultiTaskModel(nn.Module):
    def __init__(self, backbone_pth: str, taskA_labels: int = 4, taskB_labels: int = 3, dropout: float = 0.2, device: Optional[torch.device] = None):
        super().__init__()
        self.backbone = EmbeddingModel(projection_dim=256, normalize=True, device=device)
        if backbone_pth:
            self.backbone.load(backbone_pth)

        embed_dim = (
            self.backbone.projection.out_features
            if hasattr(self.backbone, 'projection') and self.backbone.projection is not None
            else self.backbone.model.config.hidden_size
        )

        # Task A: News classification
        self.taskA_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, taskA_labels)
        )
        # Task B: Sentiment Analysis
        self.taskB_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, taskB_labels)
        )

        self.device = self.backbone.device
        self.to(self.device)
    
    def forward(self, sentences):
        # Get the embeddings from the embedding model
        embeddings = self.backbone.encode(sentences)

        embeddings = embeddings.to(self.device)

        logitsA = self.taskA_head(embeddings)
        logitsB = self.taskB_head(embeddings)
        return logitsA, logitsB
    
    def predict(self, sentences):
        self.eval()
        with torch.no_grad():
            logitsA, logitsB = self.forward(sentences)
            probsA = torch.softmax(logitsA, dim=1)
            probsB = torch.softmax(logitsB, dim=1)
            predsA = torch.argmax(probsA, dim=1)
            predsB = torch.argmax(probsB, dim=1)
        return predsA, predsB
    
    def train_epoch(self,loader,optimizer,lossA,lossB):
        self.train()
        total_loss = 0.0
        for sentences, labelsA, labelsB in loader:
            optimizer.zero_grad()
            logitsA, logitsB = self(sentences)
            lA = lossA(logitsA, labelsA.to(self.device))
            lB = lossB(logitsB, labelsB.to(self.device))
            loss = lA + lB
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self,loader,lossA,lossB):
        self.eval()
        total_loss = 0.0
        correctA = correctB = total = 0
        for sentences, labelsA, labelsB in loader:
            logitsA, logitsB = self(sentences)
            lA = lossA(logitsA, labelsA.to(self.device))
            lB = lossB(logitsB, labelsB.to(self.device))
            loss = lA + lB
            total_loss += loss.item()
            predsA = torch.argmax(logitsA, dim=1)
            predsB = torch.argmax(logitsB, dim=1)
            correctA += (predsA == labelsA.to(self.device)).sum().item()
            correctB += (predsB == labelsB.to(self.device)).sum().item()
            total += labelsA.size(0)
        avg_loss = total_loss / len(loader)
        accA = correctA / total
        accB = correctB / total
        return avg_loss, accA, accB

    # Fit the model
    def fit(self,train_loader,val_loader,optimizer,lossA,lossB,epochs = 5,scheduler = None):
        history = {"train_loss": [], "val_loss": [], "val_accA": [], "val_accB": []}
        best_val_loss = float('inf')
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer, lossA, lossB)
            val_loss, val_accA, val_accB = self.eval_epoch(val_loader, lossA, lossB)
            # Save the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save("best_model")
            if scheduler:
                scheduler.step()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accA"].append(val_accA)
            history["val_accB"].append(val_accB)
            print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                  f"A_acc: {val_accA:.2%}  B_acc: {val_accB:.2%}")
        return history
    
    # Save the model
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.backbone.save(path)
        torch.save(self.taskA_head.state_dict(), f"{path}/taskA_head.pth")
        torch.save(self.taskB_head.state_dict(), f"{path}/taskB_head.pth")
        print(f"Model saved to {path}")
        return self

    # Load the model
    def load(self, path: str):
        self.backbone.load(path)
        self.taskA_head.load_state_dict(torch.load(f"{path}/taskA_head.pth", map_location=self.device))
        self.taskB_head.load_state_dict(torch.load(f"{path}/taskB_head.pth", map_location=self.device))
        self.taskA_head.to(self.device)
        self.taskB_head.to(self.device)
        return self



    # Freeze and unfreeze methods for trainging and fine-tuning different parts of the model
    def freeze_backbone(self, num: Optional[int] = None):
        model = self.backbone.model
        # Try to access encoder layers
        try:
            layers = list(model.encoder.layer)
        except AttributeError:
            layers = []
        # Freeze logic
        if num is None or not layers:
            for p in model.parameters():
                p.requires_grad = False
            print("All backbone layers frozen")
        else:
            num_to_freeze = min(num, len(layers))
            for layer in layers[-num_to_freeze:]:
                for p in layer.parameters():
                    p.requires_grad = False
            print(f"Last {num_to_freeze} backbone layers frozen")

    def unfreeze_backbone(self, num: Optional[int] = None):
        model = self.backbone.model
        try:
            layers = list(model.encoder.layer)
        except AttributeError:
            layers = []
        if num is None or not layers:
            for p in model.parameters():
                p.requires_grad = True
            print("All backbone layers unfrozen")
        else:
            num_to_unfreeze = min(num, len(layers))
            for layer in layers[-num_to_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True
            print(f"Last {num_to_unfreeze} backbone layers unfrozen")

    def freeze_projection(self):
        if hasattr(self.backbone, "projection"):
            for p in self.backbone.projection.parameters():
                p.requires_grad = False
    
    def unfreeze_projection(self):
        if hasattr(self.backbone, "projection"):
            for p in self.backbone.projection.parameters():
                p.requires_grad = True

    def freeze_taskA_head(self):
        for param in self.taskA_head.parameters():
            param.requires_grad = False
        print("Task A head frozen")

    def unfreeze_taskA_head(self):
        for param in self.taskA_head.parameters():
            param.requires_grad = True
        print("Task A head unfrozen")

    def freeze_taskB_head(self):
        for param in self.taskB_head.parameters():
            param.requires_grad = False
        print("Task B head frozen")

    def unfreeze_taskB_head(self):
        for param in self.taskB_head.parameters():
            param.requires_grad = True
        print("Task B head unfrozen")


    # metrics
    def metrics(self, loader):
        self.eval()
        total_samples = 0

        correctA = correctB = 0
        TP_A = FP_A = FN_A = 0
        TP_B = FP_B = FN_B = 0

        with torch.no_grad():
            for sentences, labelsA, labelsB in loader:
                labelsA   = labelsA.to(self.device)
                labelsB   = labelsB.to(self.device)

                predA, predB = self.predict(sentences)

                # accumulate overall accuracy counts
                correctA += (predA == labelsA).sum().item()
                correctB += (predB == labelsB).sum().item()
                total_samples += labelsA.numel()

                # for class “1” positive:
                TP_A += ((predA == 1) & (labelsA == 1)).sum().item()
                FP_A += ((predA == 1) & (labelsA == 0)).sum().item()
                FN_A += ((predA == 0) & (labelsA == 1)).sum().item()

                TP_B += ((predB == 1) & (labelsB == 1)).sum().item()
                FP_B += ((predB == 1) & (labelsB == 0)).sum().item()
                FN_B += ((predB == 0) & (labelsB == 1)).sum().item()

        # now compute final metrics
        accA = correctA / total_samples
        accB = correctB / total_samples

        precisionA = TP_A / (TP_A + FP_A) if (TP_A + FP_A) > 0 else 0
        recallA    = TP_A / (TP_A + FN_A) if (TP_A + FN_A) > 0 else 0

        precisionB = TP_B / (TP_B + FP_B) if (TP_B + FP_B) > 0 else 0
        recallB    = TP_B / (TP_B + FN_B) if (TP_B + FN_B) > 0 else 0

        return accA, accB, recallA, recallB, precisionA, precisionB
