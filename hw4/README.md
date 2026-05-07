# README - Task A: Intel Image Classification using CNNs

## Overview
This task focuses on solving a multi-class image classification problem using the Intel Image Classification Dataset obtained from Kaggle. The dataset contains natural scene images grouped into six semantic categories: buildings, forest, glacier, mountain, sea, and street. The objective is to classify each image into its corresponding class using convolutional neural networks implemented in PyTorch and to analyze the effect of model complexity and hyperparameter tuning.

## Dataset Preparation
The Intel Image Classification dataset was downloaded using the Kaggle API and organized into training and validation folders. Standard image preprocessing techniques were applied, including resizing all images to a uniform input size, converting them to tensors, and applying normalization. The dataset was then loaded into PyTorch using custom DataLoaders with shuffled mini-batches.

## Model Architectures
Two CNN architectures were developed and compared:

### 1. Lightweight CNN (~500K parameters)
A custom shallow convolutional neural network with a limited number of convolutional filters, max-pooling layers, batch normalization, and fully connected layers was designed to maintain approximately 500,000 trainable parameters. This model was intended to provide a computationally efficient baseline.

### 2. Deep CNN (~10M parameters)
A significantly deeper CNN architecture inspired by ResNet-style feature extraction was implemented with approximately 10 million parameters. This model includes more convolutional blocks and a richer feature hierarchy for improved spatial understanding.

## Hyperparameter Experiments
For each model, at least two hyperparameter configurations were tested. The experiments included:

- Adam optimizer vs SGD with momentum
- Learning rate comparison
- Different dropout rates
- Batch normalization impact

This allowed analysis of how optimization strategy and regularization influence convergence and validation performance.

## Performance Evaluation
Training and validation loss/accuracy curves were generated for each experiment. Additionally, confusion matrices were computed to visualize class-wise prediction behavior. Correctly classified and misclassified image samples were also examined to identify difficult visual similarities, particularly between glacier and mountain scenes.

## Key Observations
The lightweight CNN trained faster and required fewer computational resources but showed reduced feature discrimination on complex classes. The deeper CNN consistently achieved lower validation loss and higher classification accuracy due to its stronger representational power. Adam optimizer provided faster convergence, while higher dropout improved generalization by reducing overfitting.

## Potential Improvements
Further performance can be improved through stronger data augmentation, transfer learning with pretrained backbones such as ResNet or EfficientNet, learning rate scheduling, and longer training.

---

# README - Task B: NanoGPT Transformer Performance Improvement

## Overview
This task builds upon the provided NanoGPT transformer notebook and focuses on improving text generation performance using the tips-ML-Phenotype.txt dataset. The tokenized dataset was divided into a 90:10 train-validation split, and multiple structured modifications were applied to the baseline GPT implementation to improve language modeling capability.

## Baseline Model
The original notebook used character-level tokenization via stoi/itos, learned positional embeddings using nn.Embedding(block_size, n_embd), and fixed transformer depth and embedding size. This baseline served as the reference implementation before modifications.

## Modification A: Sub-word Tokenization using Tiktoken
The original character-level tokenizer was replaced with a sub-word tokenizer using the tiktoken library. Instead of treating each character independently, the model now processes semantically meaningful word fragments, reducing token sequence length and improving contextual understanding. This modification resulted in smoother and more coherent generated text.

## Modification B: Rotary Positional Embeddings (RoPE)
The learned positional embedding layer was replaced with Rotary Positional Embeddings. RoPE injects relative positional information directly into the attention queries and keys through rotational transformations. Compared to static positional embeddings, RoPE improves sequence ordering awareness and enhances long-context dependency modeling.

## Modification C: Hyperparameter Search
A structured hyperparameter search was performed over two major GPT architecture parameters:

- n_layer = {4, 6, 8}
- n_embd = {64, 128, 256}

This produced a total of nine experimental model configurations. Increasing n_layer improves transformer depth, while increasing n_embd expands hidden representation capacity. Final pretraining losses were recorded for each setting to analyze the trade-off between model size and performance.

## Performance Evaluation
For every experiment, training and validation losses were monitored. Generated text samples were also qualitatively compared. Models with larger embedding size and deeper transformer stacks generally produced lower pretraining loss and more contextually meaningful generated text, although at the cost of longer training time.

## Key Observations
Sub-word tokenization significantly improved semantic continuity in generated outputs. RoPE enhanced positional understanding compared to fixed embeddings. Among the hyperparameter settings, deeper models with larger embedding sizes consistently achieved the best pretraining loss, confirming that both model depth and representation width are critical to GPT performance.

## Potential Improvements
Future improvements may include larger context window, longer training iterations, weight decay tuning, learning rate scheduling, use of pretrained tokenizer vocabularies, and mixed precision training.
