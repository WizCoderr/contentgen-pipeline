from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

token = os.getenv("HUGGINGFACE_TOKEN")
model_id = "Chain-GPT/Solidity-LLM"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if hasattr(torch, "compile"):
    model = torch.compile(model)

def generate_response(prompt: str) -> str:
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=False
        )
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
    return generated_text

if __name__ == "__main__":
    print("Warming up model...")
    _ = generate_response("Warmup run")
    prompt = "What is the future of AI in content generation?"
    response = generate_response(prompt)
    print("Generated Response:", response)