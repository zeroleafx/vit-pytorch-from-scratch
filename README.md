# Vision Transformer (ViT) from Scratch in PyTorch

A from-scratch implementation of the Vision Transformer (ViT) architecture using PyTorch.

The model is trained for image classification on the Tiny ImageNet dataset.

## Architecture

The model follows the main components of Vision Transformer:

- Patch embedding using `Conv2d`
- Learnable class token
- Learnable positional embeddings
- Multi-head self-attention
- MLP / feed-forward blocks
- Transformer encoder blocks
- Classification head

For a 64×64 image with patch size 8:

    64 / 8 = 8

which produces:

    8 × 8 = 64 patches

Each patch is projected into an embedding vector before being passed through the Transformer encoder.

## Model Flow

    Input Image
    (B, 3, 64, 64)

        ↓ Patch Embedding

    (B, 64, embedding_dim)

        ↓ Add CLS Token

    (B, 65, embedding_dim)

        ↓ Add Positional Embedding

    Transformer Encoder
        ↓
    Multi-Head Self-Attention
        ↓
    MLP
        ↓
    Repeated Encoder Blocks

        ↓

    CLS Token Representation

        ↓

    Classification Head

        ↓

    (B, 200)

The final 200 logits correspond to the 200 classes in Tiny ImageNet.

## Dataset

This project uses the Tiny ImageNet dataset from Hugging Face:

    Maysee/tiny-imagenet

The dataset contains:

- 200 image classes
- 64×64 RGB images
- Training and validation splits

The Hugging Face dataset is wrapped in a custom PyTorch `Dataset` and loaded using `DataLoader`.

## Training

Loss function:

    CrossEntropyLoss

Optimizer:

    AdamW

The model is trained from scratch without pretrained weights.

Example configuration:

    Image size:       64 × 64
    Patch size:       8 × 8
    Embedding dim:    256
    Batch size:       64
    Number of classes: 200

## Project Structure

    .
    ├── models/
    │   ├── patchEmbedding.py
    │   ├── mlp.py
    │   ├── multiheadAttention.py
    │   ├── transformerEncoder.py
    │   └── vit.py
    │
    ├── data_setup.py
    ├── engine.py
    ├── train.py
    ├── utils.py
    ├── README.md
    └── .gitignore

## Running the Project

Install the required dependencies:

    pip install torch torchvision datasets pillow tqdm

Train the model:

    python train.py

## Results

Example training results:

| Metric | Result |
|---|---:|
| Training Accuracy | TODO |
| Validation Accuracy | TODO |
| Validation Loss | TODO |

## What I Learned

This project was built to better understand the internal architecture of Vision Transformers and the PyTorch training workflow.

Key concepts explored include:

- Patch embeddings
- Multi-head self-attention
- Positional embeddings
- Class tokens
- Transformer encoder blocks
- Residual connections
- Layer normalization
- AdamW optimization
- Image classification with PyTorch

## What I Will Do Next
- Fix current problem: Overfitting occurs at around 35 epochs, stuck at 40% accuracy
- More data augmentation?
- Add training and validation curves 
