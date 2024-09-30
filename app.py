import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from functools import lru_cache

# Initialize the FastAPI app
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

model_name = "uuhnnoo/gptBOT"  

@lru_cache(maxsize=1)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline('text-generation', model=model, tokenizer=tokenizer)

generator = get_model()


class QueryRequest(BaseModel):
    prompt: str

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse('static/index.html')


@app.post("/generate/")
async def generate_text(query: QueryRequest):
    response = generator(query.prompt, max_length=100, num_return_sequences=1)
    generated_text = response[0]['generated_text']
    generated_text = generated_text[len(query.prompt):].strip()
    return {"response": generated_text}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
