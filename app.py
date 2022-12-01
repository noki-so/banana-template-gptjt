from transformers import GPTJForCausalLM, GPT2Tokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    print("loading to CPU...")
    model = GPTJForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")
    else:
        raise Exception("No GPU provided!")

    tokenizer = GPT2Tokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # pop prompt, so we can use the rest of the model inputs as **kwargs in generate
    model_inputs.pop("prompt")

    # Tokenize inputs
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Run the model
    output = model.generate(input_tokens, **model_inputs)

    # Decode output tokens
    output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

    result = {"output": output_text}

    # Return the results as a dictionary
    return result
