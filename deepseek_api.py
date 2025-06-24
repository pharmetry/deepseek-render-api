# deepseek_api.py
# FastAPI server to run a lightweight code model for HTML/JS/CSS generation (Render-compatible)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Lightweight model for code generation (under 512MB RAM)
model_id = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

# Use CPU only (Render Free Tier has no GPU and limited RAM)

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512

@app.post("/generate")
async def generate(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"result": decoded}
