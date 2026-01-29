# Implementing an LLM on iPhone

A complete implementation of GPT-2 running on iOS devices for on-device inference. This project demonstrates how to convert a pre-trained PyTorch model to Apple's Core ML format and build a functional chat interface on iPhone.

## Overview

This project implements a **Large Language Model (LLM)** on an iOS device for real-time, on-device text generation. The focus is on **optimizing a pre-trained GPT-2 model** for Apple devices, converting it to the **Core ML** format, and enabling **efficient inference** without cloud dependency.

## Features

- **On-Device Inference**: Complete privacy - no data sent to the cloud
- **Model Conversion Pipeline**: PyTorch → TorchScript → Core ML
- **Model Optimization**: Quantization and palettization for reduced model size
- **Custom BPE Tokenization**: Efficient text processing matching GPT-2 standard
- **Advanced Sampling**: Supports greedy and top-k (k=40) sampling strategies
- **SwiftUI Interface**: Modern, responsive chat interface
- **Generation Control**: Start/stop generation with real-time timing metrics

## Project Structure

```
Implementing-an-LLM-on-iPhone/
├── README.md                    # Project documentation
├── Report.pdf                   # Detailed project report
├── chat.py                      # Python script to test Core ML model
├── final_conversion.py          # PyTorch → Core ML conversion script
└── chatgpt2/                    # Xcode iOS project
    └── chatgpt2/
        ├── chatgpt2App.swift    # App entry point
        ├── ContentView.swift    # SwiftUI interface
        ├── GPT2ViewModel.swift  # ViewModel for model interaction
        ├── Sources/
        │   ├── GPT2.swift              # Core inference engine
        │   ├── GPT2Tokenizer.swift     # BPE tokenization
        │   ├── GPT2ByteEncoder.swift   # Byte-pair encoding mapping
        │   ├── Math.swift              # Softmax, top-k, sampling
        │   ├── MLMultiArray+Utils.swift # Core ML array utilities
        │   └── Utils.swift             # Helper utilities
        ├── Resources/
        │   ├── gpt2-vocab.json         # GPT-2 vocabulary
        │   ├── gpt2-merges.txt         # BPE merge rules
        │   └── encoded_tokens.json     # Pre-encoded tokens
        └── Tests/                       # Unit and UI tests
```

## Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI Framework | SwiftUI | Modern iOS interface |
| ML Framework | Core ML | On-device inference |
| Model Format | .mlpackage | Apple's optimized ML format |
| Model Source | Hugging Face | Pre-trained GPT-2 weights |
| Conversion | coremltools | PyTorch → Core ML pipeline |
| Language | Swift 5.8+ | iOS development |
| Optimization | Accelerate | SIMD numerical operations |

## Setup & Installation

### Prerequisites

- **macOS** with Xcode 15+
- **Python 3.8+** (for model conversion)
- **iOS 18+** device or simulator

### 1. Clone the Repository

```sh
git clone https://github.com/Mokshesh19/Implementing-an-LLM-on-iPhone.git
cd Implementing-an-LLM-on-iPhone
```

### 2. Install Python Dependencies (For Model Conversion)

```sh
pip install torch torchvision torchaudio transformers coremltools numpy
```

### 3. Convert the Model (Optional)

If you need to regenerate the Core ML model:

```python
# Using the provided conversion script
python final_conversion.py
```

Or manually:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import coremltools as ct

# Load pre-trained GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Convert to TorchScript
model.eval()
traced_model = torch.jit.trace(model, torch.randint(0, 50257, (1, 64)))

# Convert to Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input_ids", shape=(1, 64), dtype=int)]
)
mlmodel.save("GPT2.mlpackage")
```

### 4. Run the iOS App

1. Open `chatgpt2/chatgpt2.xcodeproj` in Xcode
2. Add your converted `.mlpackage` model to the project
3. Select your target device (iPhone/iPad or Simulator)
4. Build and run (⌘+R)

## Usage

1. Launch the app on your iOS device
2. Enter a text prompt in the input field
3. Tap **Generate** to start text generation
4. View generated text with inference timing metrics
5. Tap **Stop** to cancel generation at any time

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Input    │────▶│  GPT2Tokenizer   │────▶│    Token IDs    │
└─────────────────┘     │  (BPE Encoding)  │     └────────┬────────┘
                        └──────────────────┘              │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Generated Text │◀────│  GPT2Tokenizer   │◀────│  Core ML Model  │
└─────────────────┘     │  (BPE Decoding)  │     │   (GPT-2)       │
                        └──────────────────┘     └─────────────────┘
```

### Key Components

- **GPT2.swift**: Main inference engine handling token generation loop with support for cancellation
- **GPT2Tokenizer.swift**: Implements Byte Pair Encoding (BPE) for text ↔ token conversion
- **Math.swift**: High-performance numerical operations using Apple's Accelerate framework
- **GPT2ViewModel.swift**: MVVM architecture managing model state and UI updates

## Performance Metrics

| Optimization | Storage Reduction | Notes |
|--------------|-------------------|-------|
| Quantization | -74% to -75% | INT8 weights |
| Palettization | Additional savings | k-means clustering |

### Generation Settings

- **Sequence Length**: 64 tokens (configurable)
- **Sampling Strategies**: Greedy (fast) or Top-k (k=40, diverse)
- **Threading**: Background inference, main-thread UI updates

## Testing the Model (Python)

You can test the converted model directly in Python:

```sh
python chat.py
```

## Future Improvements

- Support for larger models (GPT-2 Medium/Large)
- Neural Engine optimization for A-series chips
- Streaming token generation
- Chat history persistence
- Model caching improvements

## Resources

- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Hugging Face GPT-2](https://huggingface.co/gpt2)
- [coremltools](https://github.com/apple/coremltools)

## License

This project is for educational purposes.

## Contributors

- **Mokshesh Jain** ([@Mokshesh19](https://github.com/Mokshesh19))
