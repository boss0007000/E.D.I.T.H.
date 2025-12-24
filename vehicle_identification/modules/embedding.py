"""
Embedding and Retrieval Module
Creates embeddings for images and performs similarity search
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import pickle
import os


class EmbeddingRetrieval:
    """Handles image embeddings and similarity search"""
    
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32",
                 dimension: int = 512,
                 device: str = "cpu",
                 index_type: str = "faiss"):
        """
        Initialize embedding and retrieval system
        
        Args:
            model_name: Model for creating embeddings
            dimension: Embedding dimension
            device: Device to run on
            index_type: Index type ('faiss' or 'hnswlib')
        """
        self.model_name = model_name
        self.dimension = dimension
        self.device = torch.device(device)
        self.index_type = index_type
        self.model = None
        self.processor = None
        self.index = None
        self.database = []
    
    def load_model(self):
        """Load embedding model"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded {self.model_name} for embeddings")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Using mock embeddings for demonstration")
            self.model = None
    
    def create_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Create embedding for an image
        
        Args:
            image: Input image
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            return self._mock_embedding()
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy()[0]
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return self._mock_embedding()
    
    def _mock_embedding(self) -> np.ndarray:
        """Create mock embedding for demonstration"""
        embedding = np.random.randn(self.dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def build_index(self, database: List[Dict], embeddings: np.ndarray = None):
        """
        Build search index from database
        
        Args:
            database: List of vehicle records
            embeddings: Pre-computed embeddings (optional)
        """
        self.database = database
        
        if embeddings is None:
            print("Computing embeddings for database...")
            embeddings = []
            for record in database:
                # In a real system, would load and embed actual images
                emb = self._mock_embedding()
                embeddings.append(emb)
            embeddings = np.array(embeddings)
        
        # Build index
        if self.index_type == "faiss":
            self._build_faiss_index(embeddings)
        elif self.index_type == "hnswlib":
            self._build_hnswlib_index(embeddings)
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index"""
        try:
            import faiss
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            
            # Create index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
            self.index.add(embeddings.astype(np.float32))
            
            print(f"Built FAISS index with {len(embeddings)} vectors")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            self.index = None
    
    def _build_hnswlib_index(self, embeddings: np.ndarray):
        """Build HNSWLIB index"""
        try:
            import hnswlib
            
            # Create index
            self.index = hnswlib.Index(space='cosine', dim=self.dimension)
            self.index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
            self.index.add_items(embeddings, list(range(len(embeddings))))
            self.index.set_ef(50)
            
            print(f"Built HNSWLIB index with {len(embeddings)} vectors")
        except Exception as e:
            print(f"Error building HNSWLIB index: {e}")
            self.index = None
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar vehicles
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if self.index is None:
            return self._mock_search(top_k)
        
        try:
            if self.index_type == "faiss":
                # Normalize query
                query = query_embedding.reshape(1, -1).astype(np.float32)
                import faiss
                faiss.normalize_L2(query)
                
                # Search
                similarities, indices = self.index.search(query, top_k)
                results = [(int(idx), float(sim)) for idx, sim in zip(indices[0], similarities[0])]
            
            elif self.index_type == "hnswlib":
                query = query_embedding.astype(np.float32)
                indices, distances = self.index.knn_query(query, k=top_k)
                # Convert distance to similarity (1 - distance for cosine)
                similarities = 1 - distances[0]
                results = [(int(idx), float(sim)) for idx, sim in zip(indices[0], similarities)]
            
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return self._mock_search(top_k)
    
    def _mock_search(self, top_k: int) -> List[Tuple[int, float]]:
        """Mock search for demonstration"""
        results = []
        for i in range(min(top_k, 5)):
            results.append((i, 0.9 - i * 0.1))
        return results
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Retrieve vehicle records
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results
            
        Returns:
            List of vehicle records with similarity scores
        """
        search_results = self.search(query_embedding, top_k)
        
        retrieved = []
        for idx, score in search_results:
            if idx < len(self.database):
                record = self.database[idx].copy()
                record['similarity_score'] = score
                retrieved.append(record)
        
        return retrieved
    
    def save_index(self, index_path: str, database_path: str):
        """Save index and database to disk"""
        try:
            if self.index_type == "faiss" and self.index is not None:
                import faiss
                faiss.write_index(self.index, index_path)
            elif self.index_type == "hnswlib" and self.index is not None:
                self.index.save_index(index_path)
            
            with open(database_path, 'wb') as f:
                pickle.dump(self.database, f)
            
            print(f"Saved index and database")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self, index_path: str, database_path: str):
        """Load index and database from disk"""
        try:
            if self.index_type == "faiss" and os.path.exists(index_path):
                import faiss
                self.index = faiss.read_index(index_path)
            elif self.index_type == "hnswlib" and os.path.exists(index_path):
                import hnswlib
                self.index = hnswlib.Index(space='cosine', dim=self.dimension)
                self.index.load_index(index_path)
            
            if os.path.exists(database_path):
                with open(database_path, 'rb') as f:
                    self.database = pickle.load(f)
            
            print(f"Loaded index with {len(self.database)} records")
        except Exception as e:
            print(f"Error loading index: {e}")
