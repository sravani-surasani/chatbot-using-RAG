CODE:
# Multi-Agent RAG System with Model Context Protocol (MCP)

import os
import uuid
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import tempfile
import shutil

# Core dependencies
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Document parsing
import PyPDF2
from pptx import Presentation
import docx
import csv
from pathlib import Path

# LLM Integration (using OpenAI as example)
import openai
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MCP Message Protocol
# =============================================================================

@dataclass
class MCPMessage:
    """Model Context Protocol Message Structure"""
    sender: str
    receiver: str
    type: str
    trace_id: str
    payload: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        return cls(**data)

# Message Types
class MessageType:
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    INGESTION_REQUEST = "INGESTION_REQUEST"
    INGESTION_RESPONSE = "INGESTION_RESPONSE"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESPONSE = "RETRIEVAL_RESPONSE"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    ERROR = "ERROR"

# =============================================================================
# Message Bus (In-Memory Implementation)
# =============================================================================

class MessageBus:
    """In-memory message bus for MCP communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
        self.message_history: List[MCPMessage] = []
    
    def subscribe(self, agent_id: str, callback: callable):
        """Subscribe an agent to receive messages"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
    
    async def publish(self, message: MCPMessage):
        """Publish a message to the bus"""
        self.message_history.append(message)
        logger.info(f"Publishing message: {message.sender} -> {message.receiver} ({message.type})")
        
        # Deliver to specific receiver
        if message.receiver in self.subscribers:
            for callback in self.subscribers[message.receiver]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error delivering message to {message.receiver}: {e}")
    
    def get_message_history(self, trace_id: str = None) -> List[MCPMessage]:
        """Get message history for debugging"""
        if trace_id:
            return [msg for msg in self.message_history if msg.trace_id == trace_id]
        return self.message_history

# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.message_bus.subscribe(agent_id, self.handle_message)
    
    @abstractmethod
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        pass
    
    async def send_message(self, receiver: str, message_type: str, payload: Dict[str, Any], trace_id: str):
        """Send a message via the message bus"""
        message = MCPMessage(
            sender=self.agent_id,
            receiver=receiver,
            type=message_type,
            trace_id=trace_id,
            payload=payload
        )
        await self.message_bus.publish(message)

# =============================================================================
# Document Parsers
# =============================================================================

class DocumentParser:
    """Handles parsing of various document formats"""
    
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Parse PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Parse DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def parse_pptx(file_path: str) -> str:
        """Parse PPTX file"""
        try:
            prs = Presentation(file_path)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error parsing PPTX {file_path}: {e}")
            return ""
    
    @staticmethod
    def parse_csv(file_path: str) -> str:
        """Parse CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Error parsing CSV {file_path}: {e}")
            return ""
    
    @staticmethod
    def parse_txt(file_path: str) -> str:
        """Parse TXT/Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error parsing TXT {file_path}: {e}")
            return ""

# =============================================================================
# Ingestion Agent
# =============================================================================

class IngestionAgent(BaseAgent):
    """Handles document parsing and preprocessing"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__("IngestionAgent", message_bus)
        self.parser = DocumentParser()
        self.processed_documents: Dict[str, Dict[str, Any]] = {}
    
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        if message.type == MessageType.INGESTION_REQUEST:
            await self.process_documents(message)
    
    async def process_documents(self, message: MCPMessage):
        """Process uploaded documents"""
        try:
            documents = message.payload.get("documents", [])
            processed_docs = []
            
            for doc_info in documents:
                file_path = doc_info["file_path"]
                file_name = doc_info["file_name"]
                file_type = doc_info["file_type"]
                
                # Parse document based on type
                if file_type == "pdf":
                    text = self.parser.parse_pdf(file_path)
                elif file_type == "docx":
                    text = self.parser.parse_docx(file_path)
                elif file_type == "pptx":
                    text = self.parser.parse_pptx(file_path)
                elif file_type == "csv":
                    text = self.parser.parse_csv(file_path)
                elif file_type in ["txt", "md"]:
                    text = self.parser.parse_txt(file_path)
                else:
                    text = ""
                
                # Chunk the text
                chunks = self.chunk_text(text)
                
                doc_data = {
                    "file_name": file_name,
                    "file_type": file_type,
                    "text": text,
                    "chunks": chunks,
                    "processed_at": datetime.now().isoformat()
                }
                
                processed_docs.append(doc_data)
                self.processed_documents[file_name] = doc_data
            
            # Send response to RetrievalAgent
            await self.send_message(
                receiver="RetrievalAgent",
                message_type=MessageType.INGESTION_RESPONSE,
                payload={"processed_documents": processed_docs},
                trace_id=message.trace_id
            )
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            await self.send_message(
                receiver="CoordinatorAgent",
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                trace_id=message.trace_id
            )
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks

# =============================================================================
# Retrieval Agent
# =============================================================================

class RetrievalAgent(BaseAgent):
    """Handles embeddings and semantic retrieval"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__("RetrievalAgent", message_bus)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = None
        self.document_chunks: List[Dict[str, Any]] = []
        self.index_built = False
    
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        if message.type == MessageType.INGESTION_RESPONSE:
            await self.build_vector_store(message)
        elif message.type == MessageType.RETRIEVAL_REQUEST:
            await self.retrieve_relevant_chunks(message)
    
    async def build_vector_store(self, message: MCPMessage):
        """Build vector store from processed documents"""
        try:
            processed_docs = message.payload.get("processed_documents", [])
            all_chunks = []
            
            for doc in processed_docs:
                for chunk in doc["chunks"]:
                    chunk_data = {
                        "text": chunk,
                        "source": doc["file_name"],
                        "file_type": doc["file_type"]
                    }
                    all_chunks.append(chunk_data)
            
            if not all_chunks:
                return
            
            # Generate embeddings
            chunk_texts = [chunk["text"] for chunk in all_chunks]
            embeddings = self.embedding_model.encode(chunk_texts)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(embeddings.astype('float32'))
            
            self.document_chunks = all_chunks
            self.index_built = True
            
            logger.info(f"Vector store built with {len(all_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
    
    async def retrieve_relevant_chunks(self, message: MCPMessage):
        """Retrieve relevant chunks for a query"""
        try:
            query = message.payload.get("query", "")
            top_k = message.payload.get("top_k", 5)
            
            if not self.index_built or not query:
                await self.send_message(
                    receiver="LLMResponseAgent",
                    message_type=MessageType.RETRIEVAL_RESPONSE,
                    payload={"retrieved_context": [], "query": query},
                    trace_id=message.trace_id
                )
                return
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search vector store
            distances, indices = self.vector_store.search(
                query_embedding.astype('float32'), top_k
            )
            
            # Get relevant chunks
            relevant_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx]
                    chunk_with_score = {
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "file_type": chunk["file_type"],
                        "score": float(distances[0][i])
                    }
                    relevant_chunks.append(chunk_with_score)
            
            # Send response to LLMResponseAgent
            await self.send_message(
                receiver="LLMResponseAgent",
                message_type=MessageType.RETRIEVAL_RESPONSE,
                payload={
                    "retrieved_context": relevant_chunks,
                    "query": query
                },
                trace_id=message.trace_id
            )
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            await self.send_message(
                receiver="CoordinatorAgent",
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                trace_id=message.trace_id
            )

# =============================================================================
# LLM Response Agent
# =============================================================================

class LLMResponseAgent(BaseAgent):
    """Handles LLM query formation and response generation"""
    
    def __init__(self, message_bus: MessageBus, openai_api_key: str = None):
        super().__init__("LLMResponseAgent", message_bus)
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
    
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        if message.type == MessageType.RETRIEVAL_RESPONSE:
            await self.generate_response(message)
    
    async def generate_response(self, message: MCPMessage):
        """Generate LLM response using retrieved context"""
        try:
            query = message.payload.get("query", "")
            retrieved_context = message.payload.get("retrieved_context", [])
            
            # Build context from retrieved chunks
            context_text = ""
            sources = []
            
            for i, chunk in enumerate(retrieved_context):
                context_text += f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n\n"
                sources.append({
                    "source": chunk["source"],
                    "file_type": chunk["file_type"],
                    "text": chunk["text"][:200] + "..."
                })
            
            # Generate response
            if self.openai_client and context_text:
                response_text = await self.call_openai(query, context_text)
            else:
                response_text = self.generate_fallback_response(query, retrieved_context)
            
            # Send final response
            await self.send_message(
                receiver="CoordinatorAgent",
                message_type=MessageType.LLM_RESPONSE,
                payload={
                    "query": query,
                    "response": response_text,
                    "sources": sources,
                    "context_used": len(retrieved_context)
                },
                trace_id=message.trace_id
            )
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            await self.send_message(
                receiver="CoordinatorAgent",
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                trace_id=message.trace_id
            )
    
    async def call_openai(self, query: str, context: str) -> str:
        """Call OpenAI API"""
        try:
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_fallback_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate fallback response when OpenAI is not available"""
        if not context:
            return "I couldn't find relevant information in the uploaded documents to answer your question."
        
        response = f"Based on the uploaded documents, I found {len(context)} relevant pieces of information related to your query: '{query}'.\n\n"
        
        for i, chunk in enumerate(context[:3]):  # Show top 3 chunks
            response += f"From {chunk['source']}:\n{chunk['text'][:300]}...\n\n"
        
        return response

# =============================================================================
# Coordinator Agent
# =============================================================================

class CoordinatorAgent(BaseAgent):
    """Coordinates the entire RAG workflow"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__("CoordinatorAgent", message_bus)
        self.active_traces: Dict[str, Dict[str, Any]] = {}
    
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        if message.type == MessageType.LLM_RESPONSE:
            await self.handle_final_response(message)
        elif message.type == MessageType.ERROR:
            await self.handle_error(message)
    
    async def process_user_query(self, query: str, uploaded_files: List[Dict[str, Any]] = None) -> str:
        """Process a user query through the RAG pipeline"""
        trace_id = str(uuid.uuid4())
        
        self.active_traces[trace_id] = {
            "query": query,
            "status": "processing",
            "started_at": datetime.now().isoformat()
        }
        
        try:
            # If files are uploaded, process them first
            if uploaded_files:
                await self.send_message(
                    receiver="IngestionAgent",
                    message_type=MessageType.INGESTION_REQUEST,
                    payload={"documents": uploaded_files},
                    trace_id=trace_id
                )
                
                # Wait for ingestion and indexing to complete
                await asyncio.sleep(2)
            
            # Send retrieval request
            await self.send_message(
                receiver="RetrievalAgent",
                message_type=MessageType.RETRIEVAL_REQUEST,
                payload={"query": query, "top_k": 5},
                trace_id=trace_id
            )
            
            # Wait for response
            max_wait = 30  # 30 seconds timeout
            wait_count = 0
            
            while trace_id in self.active_traces and self.active_traces[trace_id]["status"] == "processing":
                await asyncio.sleep(0.5)
                wait_count += 1
                if wait_count > max_wait * 2:  # 0.5 second intervals
                    break
            
            if trace_id in self.active_traces:
                return self.active_traces[trace_id].get("response", "Timeout: No response received")
            
            return "Error: Failed to process query"
            
        except Exception as e:
            logger.error(f"Error in coordinator: {e}")
            return f"Error processing query: {str(e)}"
    
    async def handle_final_response(self, message: MCPMessage):
        """Handle final LLM response"""
        trace_id = message.trace_id
        if trace_id in self.active_traces:
            self.active_traces[trace_id]["status"] = "completed"
            self.active_traces[trace_id]["response"] = message.payload.get("response", "")
            self.active_traces[trace_id]["sources"] = message.payload.get("sources", [])
            self.active_traces[trace_id]["completed_at"] = datetime.now().isoformat()
    
    async def handle_error(self, message: MCPMessage):
        """Handle error messages"""
        trace_id = message.trace_id
        if trace_id in self.active_traces:
            self.active_traces[trace_id]["status"] = "error"
            self.active_traces[trace_id]["error"] = message.payload.get("error", "Unknown error")

# =============================================================================
# RAG System
# =============================================================================

class MultiAgentRAGSystem:
    """Main RAG system that orchestrates all agents"""
    
    def __init__(self, openai_api_key: str = None):
        self.message_bus = MessageBus()
        
        # Initialize agents
        self.ingestion_agent = IngestionAgent(self.message_bus)
        self.retrieval_agent = RetrievalAgent(self.message_bus)
        self.llm_agent = LLMResponseAgent(self.message_bus, openai_api_key)
        self.coordinator = CoordinatorAgent(self.message_bus)
        
        logger.info("Multi-Agent RAG System initialized")
    
    async def process_query(self, query: str, uploaded_files: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user query"""
        response = await self.coordinator.process_user_query(query, uploaded_files)
        
        # Get trace information
        trace_info = None
        for trace_id, trace_data in self.coordinator.active_traces.items():
            if trace_data.get("query") == query:
                trace_info = trace_data
                break
        
        return {
            "response": response,
            "trace_info": trace_info,
            "sources": trace_info.get("sources", []) if trace_info else []
        }

# =============================================================================
# Streamlit UI
# =============================================================================

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def get_file_type(filename: str) -> str:
    """Get file type from filename"""
    extension = filename.lower().split('.')[-1]
    return extension

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Multi-Agent RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Agent RAG System with MCP")
    st.markdown("Upload documents and ask questions using our intelligent multi-agent system!")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        # Get OpenAI API key from user or environment
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key (optional)",
            type="password",
            value=os.getenv("OPENAI_API_KEY", "")
        )
        st.session_state.rag_system = MultiAgentRAGSystem(openai_api_key)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'pptx', 'csv', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.type})")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    st.markdown("**Sources:**")
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"- {source['source']} ({source['file_type']})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process uploaded files
        uploaded_file_info = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                file_info = {
                    "file_name": uploaded_file.name,
                    "file_path": file_path,
                    "file_type": get_file_type(uploaded_file.name)
                }
                uploaded_file_info.append(file_info)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Processing your question..."):
                try:
                    # Process query through RAG system
                    result = asyncio.run(
                        st.session_state.rag_system.process_query(
                            prompt, uploaded_file_info
                        )
                    )
                    
                    response = result["response"]
                    sources = result["sources"]
                    
                    st.markdown(response)
                    
                    # Display sources
                    if sources:
                        st.markdown("**Sources:**")
                        for source in sources:
                            st.markdown(f"- {source['source']} ({source['file_type']})")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
        
        # Clean up temporary files
        for file_info in uploaded_file_info:
            try:
                temp_dir = os.path.dirname(file_info["file_path"])
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")
    
    # Debug information
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.header("🔍 Debug Information")
        
        # Message history
        if hasattr(st.session_state.rag_system, 'message_bus'):
            message_history = st.session_state.rag_system.message_bus.get_message_history()
            if message_history:
                st.sidebar.write(f"Total Messages: {len(message_history)}")
                
                # Show recent messages
                recent_messages = message_history[-2:]
                for msg in recent_messages:
                    st.sidebar.json({
                        "sender": msg.sender,
                        "receiver": msg.receiver,
                        "type": msg.type,
                        "timestamp": msg.timestamp
                    })

if __name__ == "__main__":
    main()
