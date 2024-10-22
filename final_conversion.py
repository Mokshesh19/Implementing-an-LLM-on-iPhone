import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import coremltools as ct
import coremltools.optimize as cto


class FinishMySentence(torch.nn.Module):
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


local_model_path = "/Users/moksheshjain/Desktop/final/local"

# Load the tokenizer and model from the local directory
tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
token_predictor = GPT2LMHeadModel.from_pretrained(local_model_path, torchscript=True).eval()


random_tokens = torch.randint(10000, (5,))
traced_token_predictor = torch.jit.trace(token_predictor, random_tokens)


model = FinishMySentence(model=traced_token_predictor)
scripted_model = torch.jit.script(model)


mlmodel = ct.convert(
    scripted_model,
#    traced_token_predictor,
    # Range for the sequence dimension to be between [1, 64]
    inputs=[ct.TensorType(name="input_ids", shape=(ct.RangeDim(1, 64),), dtype=np.int32)],
    minimum_deployment_target=ct.target.iOS18
)


mlmodel.save('finalgpt2.mlpackage')



