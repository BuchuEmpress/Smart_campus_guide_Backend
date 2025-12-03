# main script to load data, generate embeddings, and save results
"""
Embedding Service - Generates embeddings for campus locations
Reads from campus_locations.json and saves enriched data with embeddings
"""

import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding service with a sentence transformer model
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       'all-MiniLM-L6-v2' is fast and efficient (default)
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
    
    def create_embedding_text(self, location):
        """
        Create a rich text representation for embedding
        Combines name, type, and description for better semantic search
        
        Args:
            location: Dictionary containing location data
            
        Returns:
            String to be embedded
        """
        name = location.get('name', '')
        loc_type = location.get('type', '')
        description = location.get('description', '')
        
        # Combine fields for rich semantic representation
        embedding_text = f"{name}. {description}. Type: {loc_type}"
        return embedding_text
    
    def generate_embeddings(self, locations):
        """
        Generate embeddings for a list of locations
        
        Args:
            locations: List of location dictionaries
            
        Returns:
            List of locations with added 'embedding' field
        """
        print(f"Generating embeddings for {len(locations)} locations...")
        
        # Create texts for embedding
        texts = [self.create_embedding_text(loc) for loc in locations]
        
        # Generate embeddings in batch (faster than one-by-one)
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Add embeddings to location data
        enriched_locations = []
        for location, embedding in zip(locations, embeddings):
            location_copy = location.copy()
            # Convert numpy array to list for JSON serialization
            location_copy['embedding'] = embedding.tolist()
            location_copy['embedding_text'] = self.create_embedding_text(location)
            enriched_locations.append(location_copy)
        
        print("Embeddings generated successfully!")
        return enriched_locations
    
    def load_locations(self, filepath='data/locations/campus_locations.json'):
        """
        Load location data from JSON file
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            List of location dictionaries
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list format and dict with 'locations' key
        if isinstance(data, list):
            locations = data
        elif isinstance(data, dict) and 'locations' in data:
            locations = data['locations']
        else:
            raise ValueError("JSON format not recognized. Expected list or dict with 'locations' key")
        
        print(f"Loaded {len(locations)} locations from {filepath}")
        return locations
    
    def save_embeddings(self, enriched_locations, filepath='data/locations/campus_locations_embeddings.json'):
        """
        Save enriched location data with embeddings to JSON file
        
        Args:
            enriched_locations: List of location dictionaries with embeddings
            filepath: Path where to save the file
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(enriched_locations, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(enriched_locations)} locations with embeddings to {filepath}")
    
    def process_locations(self, 
                         input_file='data/locations/campus_locations.json',
                         output_file='data/locations/campus_locations_embeddings.json'):
        """
        Complete pipeline: Load → Generate Embeddings → Save
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file with embeddings
        """
        try:
            # Load locations
            locations = self.load_locations(input_file)
            
            # Generate embeddings
            enriched_locations = self.generate_embeddings(locations)
            
            # Save results
            self.save_embeddings(enriched_locations, output_file)
            
            print("\n✅ Processing complete!")
            print(f"📥 Input: {input_file}")
            print(f"📤 Output: {output_file}")
            
        except Exception as e:
            print(f"❌ Error during processing: {str(e)}")
            raise


# Main execution
if __name__ == "__main__":
    # Create service instance
    service = EmbeddingService()
    
    # Process locations
    service.process_locations()