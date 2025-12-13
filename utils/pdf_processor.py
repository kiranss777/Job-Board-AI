from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
import streamlit as st


class PDFProcessor:
    """Handle PDF processing and chunking"""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            
            total_pages = len(pdf_reader.pages)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, page in enumerate(pdf_reader.pages):
                text += page.extract_text()
                progress_bar.progress((i + 1) / total_pages)
                status_text.text(f"Extracting page {i + 1}/{total_pages}")
            
            progress_bar.empty()
            status_text.empty()
            
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using RecursiveCharacterTextSplitter
        This creates chunks based on separators
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        return chunks
    
    def process_pdf(self, pdf_file) -> List[str]:
        """
        Complete PDF processing pipeline:
        1. Extract text
        2. Chunk text
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_file)
        
        if not text:
            raise ValueError("No text extracted from PDF")
        
        st.info(f"Extracted {len(text)} characters from PDF")
        
        # Chunk text
        chunks = self.chunk_text(text)
        st.info(f"Created {len(chunks)} chunks")
        
        return chunks