 # main.py
from ast import main
from fastapi import FastAPI, UploadFile, Form
app = FastAPI()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
token = os.getenv("HUGGINGFACE_TOKEN")
model_id = "Chain-GPT/Solidity-LLM"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def generate_response(prompt: str) -> str:
    # Remove chat template logic, use plain prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    return generated_text
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Content Generator API"}

@app.post("/generate")
async def generate_content(text_input: str = Form(...), image: UploadFile = None):
    structured_prompt = generate_response(text_input)
    return {"prompt": structured_prompt}


app.debug = True  # Enable debug mode for development
main_app = app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(main_app, host="0.0.0.0", port=8000)