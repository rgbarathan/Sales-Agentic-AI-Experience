"""RAG Manager for document-based agent enrichment."""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import structlog

logger = structlog.get_logger()


class RAGManager:
    """
    Manages RAG (Retrieval Augmented Generation) for agents.
    
    Allows agents to retrieve relevant information from policy documents,
    SOPs, and other knowledge bases to enrich their responses.
    """
    
    def __init__(
        self,
        vectorstore_path: str = "./rag/vectorstore",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAG manager.
        
        Args:
            vectorstore_path: Path to store vector embeddings
            embedding_model: Sentence transformer model to use
        """
        self.vectorstore_path = Path(vectorstore_path)
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.vectorstore_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Agent-specific collections
        self.collections: Dict[str, chromadb.Collection] = {}
        
        logger.info("rag_manager_initialized", vectorstore_path=str(self.vectorstore_path))
    
    def create_collection(self, agent_name: str) -> chromadb.Collection:
        """
        Create or get a collection for an agent.
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            ChromaDB collection
        """
        collection_name = f"{agent_name}_knowledge"
        
        if agent_name not in self.collections:
            try:
                # Try to get existing collection first
                self.collections[agent_name] = self.client.get_collection(
                    name=collection_name
                )
                logger.info("rag_collection_loaded", agent_name=agent_name)
            except Exception:
                # Create new collection if it doesn't exist
                self.collections[agent_name] = self.client.create_collection(
                    name=collection_name,
                    metadata={"agent": agent_name}
                )
                logger.info("rag_collection_created", agent_name=agent_name)
        
        return self.collections[agent_name]
    
    def add_documents(
        self,
        agent_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to an agent's knowledge base.
        
        Args:
            agent_name: Name of the agent
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
        """
        collection = self.create_collection(agent_name)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"{agent_name}_doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or [{} for _ in documents],
            ids=ids
        )
        
        logger.info(
            "rag_documents_added",
            agent_name=agent_name,
            num_documents=len(documents)
        )
    
    def query(
        self,
        agent_name: str,
        query_text: str,
        n_results: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query an agent's knowledge base.
        
        Args:
            agent_name: Name of the agent
            query_text: Query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            Query results with documents and metadata
        """
        # Ensure collection is loaded
        collection = self.create_collection(agent_name)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        # Query collection
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_metadata
        )
        
        logger.info(
            "rag_query_executed",
            agent_name=agent_name,
            query=query_text,
            num_results=len(results.get("documents", [[]])[0])
        )
        
        return results
    
    def get_context(
        self,
        agent_name: str,
        query_text: str,
        n_results: int = 3
    ) -> str:
        """
        Get formatted context from knowledge base for agent prompts.
        
        Args:
            agent_name: Name of the agent
            query_text: Query text
            n_results: Number of results to retrieve
        
        Returns:
            Formatted context string
        """
        results = self.query(agent_name, query_text, n_results)
        
        if not results.get("documents") or not results["documents"][0]:
            return ""
        
        # Format context
        context_parts = ["# Relevant Knowledge Base Information\n"]
        
        documents = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            source = metadata.get("source", "Unknown")
            context_parts.append(f"## Source {i+1}: {source}")
            context_parts.append(doc)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def load_documents_from_directory(
        self,
        agent_name: str,
        directory: Path,
        file_extensions: List[str] = [".txt", ".md"]
    ):
        """
        Load all documents from a directory into an agent's knowledge base.
        
        Args:
            agent_name: Name of the agent
            directory: Directory containing documents
            file_extensions: File extensions to load
        """
        documents = []
        metadatas = []
        ids = []
        
        for ext in file_extensions:
            for file_path in directory.glob(f"*{ext}"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append(content)
                    metadatas.append({
                        "source": file_path.name,
                        "file_type": ext,
                        "agent": agent_name
                    })
                    ids.append(f"{agent_name}_{file_path.stem}")
                
                except Exception as e:
                    logger.error(
                        "rag_document_load_failed",
                        file=str(file_path),
                        error=str(e)
                    )
        
        if documents:
            self.add_documents(agent_name, documents, metadatas, ids)
            logger.info(
                "rag_directory_loaded",
                agent_name=agent_name,
                directory=str(directory),
                num_documents=len(documents)
            )


# Global RAG manager instance
rag_manager = RAGManager()
