 # main.py
from ast import main
from fastapi import FastAPI, UploadFile, Form
from app.services.prompt_generator import generate_response
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Content Generator API"}

@app.post("/generate")
async def generate_content(text_input: str = Form(...), image: UploadFile = None):
    prompt = f"Act like a proffesional markiting expert: {text_input}"
    structured_prompt = generate_response(text_input)
    return {"prompt": structured_prompt}


app.debug = True  # Enable debug mode for development
main_app = app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(main_app, host="0.0.0.0", port=8000)