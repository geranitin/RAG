import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global variables for model artifacts
global_model_artifacts = {}

def model_fn(model_dir):
    logger.info("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16).to(device)
    model.eval()
    # Store model artifacts in a global variable
    global_model_artifacts['tokenizer'] = tokenizer
    global_model_artifacts['model'] = model
    global_model_artifacts['device'] = device
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/json', "Only JSON input content type is supported"
    logger.info("Processing input data...")
    input_data = json.loads(request_body)
    return input_data

def predict_fn(input_data, model):
    logger.info("Generating predictions...")
    device = global_model_artifacts['device']
    tokenizer = global_model_artifacts['tokenizer']
    
    input_text = input_data['inputs']
    encoded_input = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(**encoded_input, max_length=512)
    return output_ids

def output_fn(prediction_output, accept_content_type):
    assert accept_content_type == 'application/json', "Only JSON output content type is supported"
    logger.info("Preparing the output...")
    tokenizer = global_model_artifacts['tokenizer']
    decoded_output = tokenizer.decode(prediction_output[0], skip_special_tokens=True)
    return json.dumps({"generated_text": decoded_output})

