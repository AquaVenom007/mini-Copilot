# Mini Copilot: Token-Level Python Code Completion

A token-level Python code completion system built using recurrent neural network architectures. The project explores and compares a Vanilla RNN and an LSTM, followed by an improved Residual LSTM architecture for better information flow across recurrent layers.

## Overview

Code completion can be formulated as a next-token prediction problem, where a model predicts the next token based on the sequence of tokens that precedes it.

This project studies three recurrent architectures:

- Vanilla RNN
- LSTM
- Residual LSTM

The models are evaluated using:

- **Top-5 Accuracy**
- **Perplexity (PPL)**
- **Trainable Parameters**
- **Training Time**

The progression from RNN → LSTM → Residual LSTM demonstrates how architectural changes can improve sequence modeling performance for source code.

## Dataset

The project uses the **Python subset of the CodeXGLUE Code Completion (Token-level)** dataset.

The dataset contains tokenized Python source-code sequences.

### Dataset Statistics

| Dataset | Samples |
|---------|--------:|
| Training Pool | 100,000 |
| Training Set | 80,000 |
| Validation Set | 20,000 |
| Test Set | 50,000 |

The original training split is divided into training and validation sets using an **80:20 split** with a fixed random seed of **42**.

### Vocabulary

The vocabulary is constructed using the **10,000 most frequent tokens** from the training set.

Special tokens are included for sequence processing:

- `<PAD>`
- `<UNK>`
- `<SOS>`
- `<EOS>`

Total vocabulary size:

**10,004 tokens**

### Sequence Processing

All sequences are padded or truncated to a fixed length of:

**1,000 tokens**

Tokens not present in the vocabulary are mapped to `<UNK>`, while `<PAD>` tokens are ignored during loss calculation.

# Architecture

All models use the same basic configuration to maintain a fair comparison.

| Parameter | Value |
|-----------|------:|
| Vocabulary Size | 10,004 |
| Embedding Dimension | 256 |
| Hidden Dimension | 512 |
| Recurrent Layers | 2 |
| Dropout | 0.5 |
| Sequence Length | 1,000 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Cross-Entropy Loss |

## 1. Vanilla RNN

The baseline architecture uses a two-layer vanilla Recurrent Neural Network.

### Architecture

- Embedding dimension: **256**
- RNN layers: **2**
- Hidden units: **512**
- Dropout: **0.5**
- Output layer: Linear layer over the vocabulary
- Trainable parameters: **8,612,628**

### Training

The model was trained for **1 epoch** using CUDA.

- Training time: **16m 58s**
- Training Loss: **2.161**
- Training Perplexity: **8.68**
- Validation Loss: **1.956**
- Validation Perplexity: **7.07**
- Validation Top-5 Accuracy: **81.89%**

## 2. LSTM

The second model replaces the vanilla RNN with an LSTM to improve the handling of long-range dependencies.

### Architecture

- Embedding dimension: **256**
- LSTM layers: **2**
- Hidden units: **512**
- Dropout: **0.5**
- Output layer: Linear layer over the vocabulary
- Trainable parameters: **11,371,284**

### Training

The model was trained for **1 epoch** using CUDA.

- Training time: **25m 49s**
- Training Loss: **2.534**
- Training Perplexity: **12.60**
- Validation Loss: **1.941**
- Validation Perplexity: **6.97**

## 3. Residual LSTM

The Residual LSTM introduces a residual connection between the outputs of two separately stacked LSTM layers.

The final hidden representation is obtained by combining the outputs of the two recurrent layers:

**H_final = H1 + H2**

This allows information from the earlier recurrent layer to be directly propagated to the final representation.

### Architecture

- Embedding dimension: **256**
- LSTM layers: **2**
- Hidden units: **512**
- Dropout: **0.5**
- Output layer: Linear layer over the vocabulary
- Trainable parameters: **11,371,284**

The residual connection does not introduce additional trainable parameters compared with the baseline LSTM.

### Training

The model was trained for **1 epoch** using CUDA.

- Training time: **28m 19s**
- Training Loss: **1.598**
- Training Perplexity: **4.943**
- Validation Loss: **1.462**
- Validation Perplexity: **4.313**
- Validation Top-5 Accuracy: **87.45%**

# Training Configuration

The same general training setup was used for all three models.

| Configuration | Value |
|--------------|-------|
| Device | CUDA GPU |
| Epochs | 1 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss | Cross-Entropy Loss |
| Padding Handling | Ignored in Loss |
| Gradient Clipping | Norm = 1 |
| Maximum Sequence Length | 1,000 |

# Results

The models were evaluated on the test set using Perplexity and Top-5 Accuracy.

| Model | Parameters | Test PPL | Top-5 Accuracy |
|-------|-----------:|---------:|---------------:|
| Vanilla RNN | 8,612,628 | 7.206 | 80.73% |
| LSTM | 11,371,284 | 5.515 | 83.54% |
| Residual LSTM | 11,371,284 | **4.041** | **87.41%** |

## Model Comparison

### Vanilla RNN vs LSTM

The LSTM achieves better test performance than the Vanilla RNN:

- Perplexity improves from **7.206 → 5.515**
- Top-5 Accuracy improves from **80.73% → 83.54%**

This demonstrates the benefit of LSTM-based sequence modeling for token-level code completion.

### LSTM vs Residual LSTM

The Residual LSTM provides a further improvement over the baseline LSTM:

- Perplexity improves from **5.515 → 4.041**
- Top-5 Accuracy improves from **83.54% → 87.41%**
- Trainable parameters remain unchanged at **11,371,284**

The Residual LSTM achieves a **26.72% relative improvement in Perplexity** compared with the baseline LSTM and a **3.87 percentage-point improvement in Top-5 Accuracy**.

# Key Findings

- The LSTM performs better than the Vanilla RNN on the test set.
- The Residual LSTM achieves the best overall performance.
- Residual connections improve information flow between recurrent layers.
- The Residual LSTM improves performance without increasing the number of trainable parameters compared with the baseline LSTM.
- Top-5 Accuracy reaches **87.41%** on the test set.
- Test Perplexity reaches **4.041**, the lowest among the three architectures.

# Technologies Used

- **Python**
- **PyTorch**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Hugging Face Datasets**
- **NLTK**
- **CUDA**

# Project Structure

Mini-Copilot/
├── mini_copilot.ipynb
├── best-rnn-model.pt
├── best-lstm-model.pt
├── best-residual-lstm-model.pt
└── README.md

# Future Improvements

Possible directions for further development include:

- Training for additional epochs
- Early stopping
- Hyperparameter tuning
- Larger vocabulary sizes
- Attention mechanisms
- Pretrained code embeddings
- Interactive code completion interface
- Comparison with Transformer-based architectures

# Acknowledgements

This project uses the **CodeXGLUE Code Completion (Token-level)** dataset, specifically the Python subset.

The project focuses on studying recurrent neural network architectures for token-level source-code completion and comparing their performance using standard language-model evaluation metrics.
