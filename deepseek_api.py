# deepseek_api.py
# FastAPI server to run a DeepSeek-compatible model for website code generation

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load the model
model_id = "deepseek-ai/deepseek-coder-1.3b-base"  # Use this or a smaller one if needed
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

if torch.cuda.is_available():
    model.to("cuda")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024

@app.post("/generate")
async def generate(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"result": decoded}
