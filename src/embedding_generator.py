from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates numerical vector embeddings for text chunks using
    a Sentence Transformers model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding generator by loading the specified model.

        Args:
            model_name (str): Name of the Sentence Transformer model to load.
                Defaults to 'all-MiniLM-L6-v2'.
        """
        self.model_name = model_name
        # Load and store the local sentence transformer model
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate vector embeddings for a list of text strings.

        Args:
            texts (List[str]): List of text strings to convert into embeddings.

        Returns:
            np.ndarray: NumPy array containing numerical vector embeddings.
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        # Encode texts into dense vectors and return as a NumPy array
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
