# Image Captioning with VGG16 & LSTM

This repository contains an end-to-end deep learning pipeline for generating natural language descriptions of images. By combining a Convolutional Neural Network (CNN) for visual feature extraction and a Long Short-Term Memory (LSTM) network for sequence generation, the model learns to "see" and describe scenes from the **Flickr8k** dataset.

##  Overview

The system follows an **Encoder-Decoder architecture**:

* **Encoder:** Uses a pre-trained **VGG16** (trained on ImageNet) to extract dense 4096-dimensional feature vectors from input images.
* **Decoder:** An **LSTM-based RNN** that takes image features as an initial state and generates tokens sequentially using Byte Pair Encoding (BPE).

## 🛠️ Key Features

* **Subword Tokenization:** Implements Hugging Face’s `tokenizers` library with **Byte Pair Encoding (BPE)** to handle a larger vocabulary efficiently.
* **Transfer Learning:** Leverages VGG16's `fc2` layer for high-level semantic image representation.
* **Custom Training Loop:** Features a robust data pipeline using `tf.data.Dataset` for optimized shuffling, batching, and prefetching.
* **Real-time Inference:** A greedy search prediction function that generates captions word-by-word for unseen images.

## 📊 Dataset

The model is trained on the **Flickr8k Dataset**, which consists of 8,091 images, each paired with 5 unique descriptive captions.

* **Images:** Resized to $224 \times 224$ pixels.
* **Captions:** Preprocessed to lowercase, stripped of punctuation, and wrapped in `<s>` (start) and `</s>` (end) tokens.

## 🏗️ Model Architecture

### 1. Image Encoder (CNN)

* **Base:** VGG16 (Pre-trained)
* **Extraction Layer:** `fc2` (Fully Connected Layer 2)
* **Output:** 4096-dimensional vector

### 2. Caption Decoder (RNN)

* **Embedding Layer:** Converts token IDs into 100-dimensional dense vectors.
* **Bottleneck Layer:** Reduces 4096-dim image features to 512-dim using `ELU` activation.
* **LSTM Layer:** 256 Units, initialized with image features to provide visual context.
* **Output Layer:** Softmax dense layer over the 5,435-token vocabulary.

## 📈 Performance

The model was trained for 10 epochs using the Adam optimizer and Sparse Categorical Crossentropy loss.

* **Training Accuracy:** ~83%
* **Validation Accuracy:** ~82%
* **Final Loss:** 0.75 (Training) / 0.89 (Validation)

