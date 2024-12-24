from pydantic import BaseModel

# Define the model interface
class InputData(BaseModel):
    question: str

class OutputData(BaseModel):
    response: str
    similarity: float