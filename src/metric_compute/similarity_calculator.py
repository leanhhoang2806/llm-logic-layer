import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:
    def calculate_similarity(text1: str, text2: str, model_name = 'all-MiniLM-L6-v2') -> float:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load the pre-trained model and move it to the appropriate device
        model = SentenceTransformer(model_name, device=device)
        
        # Generate embeddings for the input texts and move to CPU for similarity calculation
        embeddings = model.encode([text1, text2], device=device)

        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]