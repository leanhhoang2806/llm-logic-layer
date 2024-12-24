from metric_compute.similarity_calculator import SimilarityCalculator

similarity_calculator = SimilarityCalculator.calculate_similarity

# Example Usage
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "The quick brown fox jumps over the lazy dog."
similarity_score = similarity_calculator(text1, text2)
print(f"Expect Similarity Score to be close to be 1 and get : {similarity_score:.2f}")


text1 = "The car drives through the city at night."
text2 = "The city lights brighten up the streets at night."
similarity_score = similarity_calculator(text1, text2)
print(f"Expect Similarity Score to be close to be 0.5 and get : {similarity_score:.2f}")


text1 = "The quick brown fox jumps over the lazy dog."
text2 = "I enjoy eating pizza on weekends."
similarity_score = similarity_calculator(text1, text2)
print(f"Expect Similarity Score to be close to be 0.5 and get : {similarity_score:.2f}")