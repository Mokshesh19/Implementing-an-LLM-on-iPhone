#!/usr/bin/env python3
"""
Convert GPT-2 PyTorch model to Core ML format.

Usage:
    python final_conversion.py --model_path gpt2 --output finalgpt2.mlpackage
    python final_conversion.py --model_path /path/to/local/model --output output.mlpackage
"""

import os
import argparse
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import coremltools as ct
import coremltools.optimize as cto

# Configuration constants
DEFAULT_MODEL_PATH = "gpt2"  # Hugging Face model ID or local path
DEFAULT_OUTPUT_PATH = "finalgpt2.mlpackage"
DEFAULT_EOS_TOKEN_ID = 198
DEFAULT_MAX_SEQ_LENGTH = 64


class FinishMySentence(torch.nn.Module):
    """
    Wrapper module for GPT-2 that generates tokens until EOS.
    """
    def __init__(self, model=None, eos=198):
        super(FinishMySentence, self).__init__()
        self.eos = torch.tensor([eos])
        self.next_token_predictor = model.eval()
        self.default_token = torch.tensor([0])

    def forward(self, x):
        sentence = x
        token = self.default_token
        while token != self.eos:
            predictions, _ = self.next_token_predictor(sentence)
            token = torch.argmax(predictions[-1, :], dim=0, keepdim=True)
            sentence = torch.cat((sentence, token), 0)

        return sentence


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert GPT-2 to Core ML')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.environ.get('GPT2_MODEL_PATH', DEFAULT_MODEL_PATH),
        help='Path to GPT-2 model (Hugging Face ID or local path)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=os.environ.get('OUTPUT_PATH', DEFAULT_OUTPUT_PATH),
        help='Output path for Core ML model (.mlpackage)'
    )
    parser.add_argument(
        '--eos_token_id',
        type=int,
        default=DEFAULT_EOS_TOKEN_ID,
        help='End of sequence token ID'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help='Maximum sequence length for the model'
    )
    parser.add_argument(
        '--ios_target',
        type=str,
        default='iOS18',
        choices=['iOS15', 'iOS16', 'iOS17', 'iOS18'],
        help='Minimum iOS deployment target'
    )
    return parser.parse_args()


def get_ios_target(target_str):
    """Convert iOS target string to coremltools target."""
    targets = {
        'iOS15': ct.target.iOS15,
        'iOS16': ct.target.iOS16,
        'iOS17': ct.target.iOS17,
        'iOS18': ct.target.iOS18,
    }
    return targets.get(target_str, ct.target.iOS18)


def main():
    """Main conversion function."""
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Output will be saved to: {args.output}")

    # Load tokenizer and model
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        token_predictor = GPT2LMHeadModel.from_pretrained(
            args.model_path,
            torchscript=True
        ).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: For Hugging Face models, use 'gpt2', 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'")
        return 1

    print("Model loaded successfully")

    # Create sample input for tracing
    # Use a representative sample of tokens
    sample_text = "Hello, how are you?"
    sample_tokens = tokenizer.encode(sample_text)
    random_tokens = torch.tensor(sample_tokens, dtype=torch.long)

    print(f"Tracing model with sample input of length {len(sample_tokens)}...")

    # Trace the token predictor
    try:
        traced_token_predictor = torch.jit.trace(token_predictor, random_tokens)
    except Exception as e:
        print(f"Error tracing model: {e}")
        return 1

    # Wrap in FinishMySentence module
    model = FinishMySentence(model=traced_token_predictor, eos=args.eos_token_id)
    scripted_model = torch.jit.script(model)

    print(f"Converting to Core ML (target: {args.ios_target})...")

    # Convert to Core ML
    try:
        mlmodel = ct.convert(
            scripted_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=(ct.RangeDim(1, args.max_seq_length),),
                    dtype=np.int32
                )
            ],
            minimum_deployment_target=get_ios_target(args.ios_target)
        )
    except Exception as e:
        print(f"Error converting model: {e}")
        return 1

    # Save the model
    print(f"Saving model to: {args.output}")
    try:
        mlmodel.save(args.output)
    except Exception as e:
        print(f"Error saving model: {e}")
        return 1

    print("Conversion completed successfully!")
    print(f"\nModel saved to: {args.output}")
    print(f"  - Input: input_ids (shape: 1-{args.max_seq_length})")
    print(f"  - iOS target: {args.ios_target}")

    return 0


if __name__ == "__main__":
    exit(main())
