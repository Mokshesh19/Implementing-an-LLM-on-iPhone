# Implementing an LLM on iOS for on device Inference

## Overview
This project implements a **Large Language Model (LLM)** on an iOS device for real-time, on-device text generation. The focus is on **optimizing a pre-trained GPT-2 model** for Apple devices, converting it to the **Core ML** format, and enabling **efficient inference**. The project leverages **TorchScript**, **Hugging Face Transformers**, and **Swift** to create a smooth and privacy-focused AI experience on mobile devices.

## Features
- **Model Conversion & Optimization**: Converts GPT-2 to Core ML with quantization and palettization.
- **On-Device Inference**: No cloud dependency for text generation.
- **Custom Tokenization**: Efficiently processes text input.
- **Advanced Sampling Techniques**: Supports greedy and top-k sampling for high-quality text generation.
- **Performance Optimization**: Reduces model size while balancing accuracy and speed.

## Project Structure
```
📂 project-root
 ├── 📂 model_conversion      # Scripts for GPT-2 to Core ML conversion
 ├── 📂 optimization          # Quantization, pruning, and palettization scripts
 ├── 📂 ios_app              # Swift-based iOS app implementation
 ├── 📂 evaluation           # Performance tests and benchmarks
 ├── README.md              # Project documentation
```

## Setup & Installation
### 1. Clone the Repository
```sh
git clone https://github.com/Mokshesh19/Implementing-an-LLM-on-Iphone
```

### 2. Install Dependencies
#### Python Dependencies (For Model Conversion & Optimization)
```sh
pip install torch torchvision torchaudio transformers coremltools numpy
```

#### Swift & iOS Requirements
- **Xcode 15+**
- **Swift 5.8+**
- **Core ML Tools**

## Model Conversion & Optimization
1. Load a pre-trained GPT-2 model:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
```
2. Convert to TorchScript:
```python
import torch
traced_model = torch.jit.trace(model, torch.rand(1, 64))
```
3. Convert to Core ML:
```python
import coremltools as ct
mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(name="input_ids", shape=(1, 64), dtype=int)])
mlmodel.save("GPT2.mlpackage")
```

## Running the iOS App
1. Open `ios_app` in **Xcode**.
2. Build and run on an iPhone or iPad.
3. Input text, press **Generate**, and see real-time text generation!

## Performance Metrics
| Model Size   | Storage Reduction | RAM Usage | Inference Speed |
|-------------|------------------|-----------|----------------|
| GPT-2 Small | **-74%**         | **+500MB**| **Faster** |
| GPT-2 XL    | **-75%**         | **+6GB**  | **1.5x Faster** |

## Future Improvements
- Support for **GPT-3 & beyond**
- More efficient **on-device memory management**
- Integration with **SwiftUI** for improved UX

## Contributors
- **Mokshesh Jain** ([@Mokshesh19](https://github.com/Mokshesh19))



