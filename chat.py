import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import coremltools as ct

# Load the tokenizer and Core ML model
model_path = '/Users/moksheshjain/desktop/final/.mlpackage'
mlmodel = ct.models.MLModel(model_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # Subtracting max for numerical stability
    return exps / np.sum(exps)

def top_k_sampling(probs, k=50):
    top_k_indices = np.argsort(probs)[-k:]  # Get top-k token indices
    top_k_probs = probs[top_k_indices]  # Get probabilities of top-k tokens
    top_k_probs = top_k_probs / np.sum(top_k_probs)  # Normalize the probabilities
    next_token = np.random.choice(top_k_indices, p=top_k_probs)  # Sample the next token from the top-k
    return next_token

def generate_response(mlmodel, input_text, eos_token_id=198, temperature=1.2 , k=50, p=0.9):
    # Encode the input text
#    input_ids = tokenizer.encode(input_text, return_tensors="pt").numpy().astype(np.int32).flatten()
    input_ids = input_text.numpy().astype(np.int32).flatten()

    # Initialize sentence with input_ids
    sentence = list(input_ids)

    while True:
        # Prepare input for Core ML model
        coreml_input = {'input_ids': np.array(sentence, dtype=np.int32)}

        # Predict next token
        prediction_dict = mlmodel.predict(coreml_input)

        # Extract logits from the correct layer (linear_0 in this case)
        logits = prediction_dict['linear_0']

        # Apply softmax
        probs = softmax(logits[-1])

        # Sample the next token using top-k sampling
        next_token = top_k_sampling(probs, k=k)

        # Append the token to the sentence
        sentence.append(next_token)

        # Decode the token to a word
        word = tokenizer.decode(next_token, skip_special_tokens=True)
        print("Generated word:", word)

        # Break if EOS token is generated or max length is reached
        if next_token == eos_token_id or len(sentence) >= 64:
            break

    return tokenizer.decode(sentence, skip_special_tokens=True)

# Chatbot loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    input_text = tokenizer.encode(user_input, return_tensors="pt").squeeze(0)
    print("user_input: ",type(user_input))
    print("user_input: ",user_input)
    response = generate_response(mlmodel, input_text)
    print(f"GPT-2: {response}")
