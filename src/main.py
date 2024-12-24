from fastapi import FastAPI, HTTPException
from models.api_model import InputData, OutputData
from metric_compute.similarity_calculator import SimilarityCalculator
import ray
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from exceptions.exception_handler import exception_handler


smilarity_calculator = SimilarityCalculator.calculate_similarity

# Initialize FastAPI
app = FastAPI()

# Ray initialization
ray.init()

# Simulating a small LLM model using Ray actors
@ray.remote
class SmallLLMModel:
    def __init__(self):
        self.responses = {
            "hello": "Hi there!",
            "how are you?": "I'm just a program, but thanks for asking!",
            "what is your name?": "I'm a small LLM model running on Ray."
        }
        self.vectorizer = CountVectorizer()

    def generate_response(self, question: str) -> str:
        return self.responses.get(question.lower(), "I don't understand that question.")

    def calculate_similarity(self, question: str, response: str) -> float:
        vectors = self.vectorizer.fit_transform([question, response]).toarray()
        similarity = np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
        return similarity

# Deploy the model actor
llm_model = SmallLLMModel.remote()

@exception_handler
@app.post("/ask", response_model=OutputData)
async def ask_question(input_data: InputData):
    response = await llm_model.generate_response.remote(input_data.question)
    similarity = smilarity_calculator(input_data.question, response)
    return {"response": response, "similarity": similarity}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
