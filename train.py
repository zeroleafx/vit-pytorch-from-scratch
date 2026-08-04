import torch
from torch import nn
from data_setup import create_dataloaders
from engine import train
from utils import save_model
from models.vit import ViT

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

def main():
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      train_loader, valid_loader, class_names = create_dataloaders(batch_size=BATCH_SIZE)

      model = ViT()

      loss_fn = nn.CrossEntropyLoss()
      optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4, weight_decay=0.05)
      train(model=model,
            train_dataloader=train_loader,
            test_dataloader=valid_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=EPOCHS,
            device=device
            )

      save_model(model=model,
            target_dir="models",
            model_name="vit_tiny_imagenet.pth"
            )

if __name__ == "__main__":
    main()