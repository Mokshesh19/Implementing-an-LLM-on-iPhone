#!/usr/bin/env python3
"""
Chat script for testing the Core ML GPT-2 model.

Usage:
    python chat.py --model_path /path/to/model.mlpackage
    python chat.py  # Uses MODEL_PATH environment variable or default
"""

import os
import argparse
import torch
import numpy as np
from transformers import GPT2Tokenizer
import coremltools as ct

# Configuration constants
DEFAULT_MODEL_PATH = './finalgpt2.mlpackage'
DEFAULT_EOS_TOKEN_ID = 198
DEFAULT_TEMPERATURE = 1.2
DEFAULT_TOP_K = 50
DEFAULT_MAX_LENGTH = 64


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Chat with GPT-2 Core ML model')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.environ.get('MODEL_PATH', DEFAULT_MODEL_PATH),
        help='Path to the Core ML model (.mlpackage)'
    )
    parser.add_argument(
        '--eos_token_id',
        type=int,
        default=DEFAULT_EOS_TOKEN_ID,
        help='End of sequence token ID'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=DEFAULT_TOP_K,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help='Maximum sequence length'
    )
    return parser.parse_args()


def softmax(logits):
    """Apply softmax to logits with numerical stability."""
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def top_k_sampling(probs, k=50):
    """Sample from top-k most probable tokens."""
    top_k_indices = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_indices]
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    next_token = np.random.choice(top_k_indices, p=top_k_probs)
    return next_token


def generate_response(mlmodel, tokenizer, input_ids, eos_token_id=198, k=50, max_length=64):
    """
    Generate a response from the model.

    Args:
        mlmodel: The Core ML model
        tokenizer: The GPT-2 tokenizer
        input_ids: Input token IDs as numpy array
        eos_token_id: End of sequence token ID
        k: Top-k sampling parameter
        max_length: Maximum sequence length

    Returns:
        Generated text string
    """
    sentence = list(input_ids.flatten().astype(np.int32))
    max_iterations = max_length * 2  # Safety limit

    for _ in range(max_iterations):
        # Prepare input for Core ML model
        coreml_input = {'input_ids': np.array(sentence, dtype=np.int32)}

        # Predict next token
        try:
            prediction_dict = mlmodel.predict(coreml_input)
        except Exception as e:
            print(f"Prediction error: {e}")
            break

        # Extract logits
        if 'linear_0' not in prediction_dict:
            print("Error: 'linear_0' not found in model output")
            print(f"Available keys: {list(prediction_dict.keys())}")
            break

        logits = prediction_dict['linear_0']

        # Apply softmax and sample
        probs = softmax(logits[-1])
        next_token = top_k_sampling(probs, k=k)

        # Append token
        sentence.append(int(next_token))

        # Decode and print the generated word
        word = tokenizer.decode(next_token, skip_special_tokens=True)
        print(f"Generated: {word}", end=' ', flush=True)

        # Check stopping conditions
        if next_token == eos_token_id or len(sentence) >= max_length:
            break

    print()  # New line after generation
    return tokenizer.decode(sentence, skip_special_tokens=True)


def main():
    """Main chat loop."""
    args = parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at '{args.model_path}'")
        print("Please provide a valid model path using --model_path or MODEL_PATH environment variable")
        return 1

    print(f"Loading model from: {args.model_path}")

    try:
        mlmodel = ct.models.MLModel(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(f"\nGPT-2 Chat (top_k={args.top_k}, max_length={args.max_length})")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        # Encode input
        input_ids = tokenizer.encode(user_input, return_tensors="pt").squeeze(0).numpy()

        # Generate response
        print("GPT-2: ", end='')
        response = generate_response(
            mlmodel,
            tokenizer,
            input_ids,
            eos_token_id=args.eos_token_id,
            k=args.top_k,
            max_length=args.max_length
        )
        print(f"\nFull response: {response}\n")

    return 0


if __name__ == "__main__":
    exit(main())
