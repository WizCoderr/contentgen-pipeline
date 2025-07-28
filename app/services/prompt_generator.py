from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
token = os.getenv("HUGGINGFACE_TOKEN")
model_id = "Chain-GPT/Solidity-LLM"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Explicitly set to eval mode

def generate_response(prompt: str) -> str:
    with torch.no_grad():  # Disable gradient tracking for faster inference
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


if __name__ == "__main__":
    prompt = "What is the future of AI in content generation?"
    response = generate_response(prompt)
    print("Generated Response:", response)