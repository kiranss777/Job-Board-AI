from typing import List, Dict, Any
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config import Config
import streamlit as st
import time


class VectorStore:
    """Handle Pinecone vector store operations"""
    
    def __init__(self, namespace: str = "default"):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        self.namespace = namespace  # Use namespace to organize vectors
        
        # Initialize SentenceTransformer model
        with st.spinner("Loading embedding model..."):
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Connect to existing index
        try:
            self.index = self.pc.Index(
                name=self.index_name,
                host=Config.PINECONE_HOST
            )
            
            # Verify index dimension matches config
            stats = self.index.describe_index_stats()
            index_dimension = stats.get('dimension')
            if index_dimension and index_dimension != Config.EMBEDDING_DIMENSION:
                st.warning(
                    f"⚠️ Dimension mismatch: Index has {index_dimension}D but model produces {Config.EMBEDDING_DIMENSION}D. "
                    f"Please recreate the Pinecone index with dimension={Config.EMBEDDING_DIMENSION}"
                )
                
        except Exception as e:
            st.error(f"Error connecting to Pinecone index: {str(e)}")
            raise
    
    def clear_index(self):
        """Clear all vectors from the namespace"""
        try:
            # Delete all vectors in the current namespace
            self.index.delete(delete_all=True, namespace=self.namespace)
            st.info(f"Pinecone namespace '{self.namespace}' cleared successfully")
            time.sleep(1)  # Give time for deletion to propagate
        except Exception as e:
            # If namespace doesn't exist, that's fine - it means it's already empty
            if "Namespace not found" in str(e) or "404" in str(e):
                st.info(f"Namespace '{self.namespace}' is empty or doesn't exist yet")
            else:
                st.error(f"Error clearing index: {str(e)}")
                raise
    
    def upsert_documents(self, chunks: List[str], source: str = "document") -> int:
        """
        Create embeddings for chunks and upsert to Pinecone with retry logic
        Returns the number of chunks uploaded
        """
        try:
            vectors = []
            
            # Create embeddings for each chunk
            st.info(f"Creating embeddings for {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks):
                # Generate embedding using SentenceTransformer
                embedding = self.model.encode(chunk).tolist()
                
                # Verify dimension
                if len(embedding) != Config.EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Embedding dimension {len(embedding)} doesn't match expected {Config.EMBEDDING_DIMENSION}"
                    )
                
                # Create vector with metadata
                vector = {
                    "id": f"{source}-chunk_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "chunk_index": i,
                        "source": source
                    }
                }
                vectors.append(vector)
            
            # Upsert in batches with retry logic
            batch_size = Config.BATCH_SIZE
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                # Retry logic
                for attempt in range(Config.MAX_RETRIES):
                    try:
                        self.index.upsert(vectors=batch, namespace=self.namespace)
                        status_text.text(f"Uploaded batch {batch_num}/{total_batches}")
                        progress_bar.progress(batch_num / total_batches)
                        break
                    except Exception as e:
                        if attempt < Config.MAX_RETRIES - 1:
                            wait_time = 2 ** attempt
                            st.warning(f"Batch {batch_num} failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            st.error(f"Batch {batch_num} failed after {Config.MAX_RETRIES} attempts: {str(e)}")
                            raise
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"Successfully uploaded {len(chunks)} chunks to Pinecone (namespace: {self.namespace})")
            return len(chunks)
            
        except Exception as e:
            st.error(f"Error upserting documents: {str(e)}")
            raise
    
    def query(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant chunks
        Returns list of relevant text chunks with scores
        """
        if top_k is None:
            top_k = Config.TOP_K
        
        try:
            # Generate query embedding using SentenceTransformer
            query_embedding = self.model.encode(query_text).tolist()
            
            # Query Pinecone with namespace
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            
            # Extract relevant chunks
            relevant_chunks = []
            for match in results.get('matches', []):
                relevant_chunks.append({
                    'text': match['metadata']['text'],
                    'score': match['score'],
                    'chunk_index': match['metadata']['chunk_index']
                })
            
            return relevant_chunks
            
        except Exception as e:
            st.error(f"Error querying vector store: {str(e)}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        try:
            stats = self.index.describe_index_stats()
            
            # Get namespace-specific stats if available
            namespace_stats = stats.get('namespaces', {}).get(self.namespace, {})
            
            return {
                'total_vector_count': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 'N/A'),
                'namespace_vector_count': namespace_stats.get('vector_count', 0),
                'current_namespace': self.namespace
            }
        except Exception as e:
            st.error(f"Error getting index stats: {str(e)}")
            return {}
    
    def list_namespaces(self) -> List[str]:
        """List all namespaces in the index"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = list(stats.get('namespaces', {}).keys())
            return namespaces
        except Exception as e:
            st.error(f"Error listing namespaces: {str(e)}")
            return []