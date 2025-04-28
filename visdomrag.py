import os
import json
import torch
import pandas as pd
import numpy as np
import time
import logging
import argparse
import re
import uuid
import csv
from tqdm import tqdm
from io import BytesIO
from pdf2image import convert_from_path
import base64
import requests
from PIL import Image
import gc
from difflib import SequenceMatcher
import PyPDF2
import pytesseract
import traceback

# Optional imports based on selected models
try:
    import google.generativeai as genai
except ImportError:
    pass

# For embeddings and retrieval
try:
    import chromadb
    import chromadb.utils.embedding_functions as embedding_functions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("visdmrag.log"), logging.StreamHandler()]
)
logger = logging.getLogger("VisDoMRAG")

qa_prompts = {
    "feta_tab": "You are a Wikipedia editor. Answer the question with a single, well-formed, factual sentence.",
    "paper_tab": "You are a research scientist. Answer the question with a concise technical phrase.",
    "scigraphqa": "You are a scientific researcher. Answer the question in 1-2 clear, evidence-based sentences.",
    "slidevqa": "You are a presentation expert. Provide the exact answer to the question as it would appear on a slide. Be direct and precise.",
    "spiqa": "You are a scientific paper author. Answer the question in 1-3 authoritative sentences."
}

class VisDoMRAG:
    def __init__(self, config):
        """
        Initialize the VisDoMRAG pipeline.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.data_dir = config["data_dir"]
        self.output_dir = config["output_dir"]
        self.llm_model = config["llm_model"]
        self.vision_retriever = config["vision_retriever"]
        self.text_retriever = config["text_retriever"]
        self.top_k = config.get("top_k", 5)
        self.api_keys = config.get("api_keys", {})
        self.chunk_size = config.get("chunk_size", 3000)
        self.chunk_overlap = config.get("chunk_overlap", 300)
        self.force_reindex = config.get("force_reindex", False)
        self.qa_prompt = config.get("qa_prompt", "Answee the question objectively based on the context provided.")
        
        # NEW: Accept CSV path directly from config
        self.dataset_csv = config.get("csv_path")
        if not self.dataset_csv:
            # Fallback to old behavior if csv_path not provided
            self.dataset_csv = f"{self.data_dir}/{os.path.basename(self.data_dir)}.csv"
        
        # Setup output directories
        os.makedirs(f"{self.output_dir}/{self.llm_model}_vision", exist_ok=True)
        os.makedirs(f"{self.output_dir}/{self.llm_model}_text", exist_ok=True)
        os.makedirs(f"{self.output_dir}/{self.llm_model}_visdmrag", exist_ok=True)
        
        # Create retrieval directories
        os.makedirs(f"{self.data_dir}/retrieval", exist_ok=True)
        
        # Initialize LLM
        self._initialize_llm()
        
        # Load dataset
        logger.info(f"Loading dataset from {self.dataset_csv}")
        if not os.path.exists(self.dataset_csv):
            raise FileNotFoundError(f"CSV file not found: {self.dataset_csv}")
        self.df = pd.read_csv(self.dataset_csv)
        
        # Initialize document cache
        self.document_cache = {}
        
        # Initialize retrieval resources
        self._initialize_retrieval_resources()
        
    def _initialize_llm(self):
        """Initialize the LLM based on the selected model."""
        if self.llm_model == "gemini":
            if not self.api_keys.get("gemini"):
                raise ValueError("Gemini API key is required")
            genai.configure(api_key=self.api_keys["gemini"])
            self.llm = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Initialized Gemini model")
        
        elif self.llm_model == "gpt4":
            if not self.api_keys.get("openai"):
                raise ValueError("OpenAI API key is required")
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_keys["openai"])
            logger.info("Initialized GPT-4 (via OpenAI client)")
        
        elif self.llm_model == "qwen":
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                from qwen_vl_utils import process_vision_info
                
                self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
                min_pixels = 256*28*28
                max_pixels = 640*28*28
                self.qwen_processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct", 
                    min_pixels=min_pixels, 
                    max_pixels=max_pixels
                )
                self.process_vision_info = process_vision_info
                logger.info("Initialized Qwen2-VL model")
            except ImportError:
                raise ImportError("Required packages for Qwen not found. Install transformers and qwen_vl_utils.")
        else:
            raise ValueError(f"Unsupported LLM model: {self.llm_model}")
    
    def _initialize_retrieval_resources(self):
        """Initialize resources needed for retrieval."""
        # Check if we need to compute visual embeddings
        self.vision_retrieval_file = f"{self.data_dir}/retrieval/retrieval_{self.vision_retriever}.csv"
        
        # Check if we need to compute textual embeddings
        self.text_retrieval_file = f"{self.data_dir}/retrieval/retrieval_{self.text_retriever}.csv"
        
        if self.vision_retriever in ["colpali", "colqwen"]:
            if self.vision_retriever == "colpali":
                try:
                    from colpali_engine.models import ColPali, ColPaliProcessor
                    logger.info("Loading ColPali model for visual indexing")
                    self.vision_model = ColPali.from_pretrained(
                        "vidore/colpali-v1.2", 
                        torch_dtype=torch.bfloat16, 
                        device_map="cuda"
                    ).eval()
                    self.vision_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
                except ImportError:
                    raise ImportError("ColPali models not found. Please install colpali_engine.")
            elif self.vision_retriever == "colqwen":
                try:
                    from colpali_engine.models import ColQwen2, ColQwen2Processor
                    logger.info("Loading ColQwen model for visual indexing")
                    self.vision_model = ColQwen2.from_pretrained(
                        "vidore/colqwen2-v0.1", 
                        torch_dtype=torch.bfloat16, 
                        device_map="cuda"
                    ).eval()
                    self.vision_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
                except ImportError:
                    raise ImportError("ColPali/ColQwen models not found. Please install colpali_engine.")
        else:
            raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")
    
        if self.text_retriever == "bm25":
            # No model needed for BM25
            pass
        elif self.text_retriever in ["minilm", "mpnet", "bge"]:
            # Map text_retriever to actual model names
            model_map = {
                "minilm": "sentence-transformers/all-MiniLM-L6-v2",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
                "bge": "BAAI/bge-base-en-v1.5"
            }
            
            # Load sentence transformer model
            self.text_model_name = model_map[self.text_retriever]
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.st_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.text_model_name, device=self.device
            )
        else:
            raise ValueError(f"Unsupported text retriever: {self.text_retriever}")

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file using OCR if needed.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of text from each page
        """
        try:
            # First try regular PDF extraction
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file, strict=False)
                pages = [page.extract_text() for page in reader.pages]
                
            # If any page has no text, use OCR
            if any(not page.strip() for page in pages):
                logger.info(f"Using OCR for {pdf_path} as some pages have no text")
                pages = []
                pdf_images = convert_from_path(pdf_path)
                for page_num, page_img in enumerate(pdf_images):
                    text = pytesseract.image_to_string(page_img)
                    pages.append(f"--- Page {page_num + 1} ---\n{text}\n")
            
            return pages
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            traceback.print_exc()
            return []
    
    def split_text(self, text):
        """
        Split text into chunks.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_text(text)
    
    def cache_documents(self):
        """
        Cache document content for all PDFs in the dataset.
        
        Returns:
            dict: Dictionary mapping document IDs to text content
        """
        logger.info("Caching document content")
        
        try:
            # Extract unique document IDs from the dataset
            unique_docs = set()
            for _, row in self.df.iterrows():
                try:
                    docs = eval(row['documents']) if 'documents' in row else []
                    unique_docs.update(docs)
                except:
                    # Handle case where 'documents' field is not a valid list
                    traceback.print_exc()
                    pass
                
                if 'doc_path' in row:
                    doc_path = row['doc_path']
                    if isinstance(doc_path, str) and doc_path.strip():
                        unique_docs.add(os.path.basename(doc_path).split('.')[0])
            
            # Cache content for each document
            cache = {}
            pdf_dir = os.path.join(self.data_dir, "docs")
            
            for doc_id in tqdm(unique_docs, desc="Caching documents"):
                # Try different possible filename formats
                possible_paths = [
                    os.path.join(pdf_dir, doc_id),
                    os.path.join(pdf_dir, f"{doc_id}.pdf"),
                    os.path.join(pdf_dir, f"{doc_id.ljust(10, '0')}.pdf"),
                    os.path.join(pdf_dir, f"{doc_id.split('_')[0]}.pdf")
                ]
                
                for pdf_path in possible_paths:
                    if os.path.exists(pdf_path):
                        cache[doc_id] = self.extract_text_from_pdf(pdf_path)
                        break
                else:
                    logger.warning(f"No PDF file found for document {doc_id}")
            
            self.document_cache = cache
            logger.info(f"Cached content for {len(cache)} documents")
            return cache
            
        except Exception as e:
            logger.error(f"Error caching documents: {str(e)}")
            traceback.print_exc()
            return {}
    
    def identify_document_and_page(self, chunk):
        """
        Identify which document and page a chunk belongs to.
        
        Args:
            chunk (str): Text chunk
            
        Returns:
            tuple: (arxiv_id, page_num)
        """
        max_ratio = 0
        best_match = (None, None)
        
        for arxiv_id, pages in self.document_cache.items():
            for page_num, page_text in enumerate(pages):
                ratio = SequenceMatcher(None, chunk, page_text).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_match = (arxiv_id, page_num)
        
        return best_match
    
    def build_visual_index(self):
        """
        Build visual embedding index for all PDFs in the dataset using multi-vector scoring.
        
        Returns:
            bool: Success status
        """
        logger.info(f"Building visual index using {self.vision_retriever}")
        
        try:
            pdf_dir = os.path.join(self.data_dir, "docs")
            output_dir = os.path.join(self.data_dir, "visual_embeddings")
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract unique document IDs from the dataset
            unique_docs = set()
            for _, row in self.df.iterrows():
                try:
                    docs = eval(row['documents']) if 'documents' in row else []
                    unique_docs.update(docs)
                except:
                    traceback.print_exc()
                    pass
                
                if 'doc_path' in row:
                    doc_path = row['doc_path']
                    if isinstance(doc_path, str) and doc_path.strip():
                        unique_docs.add(os.path.basename(doc_path))
            
            # Get list of PDF files based on unique_docs
            pdf_files = []
            for doc_id in unique_docs:
                if doc_id.endswith('.pdf'):
                    pdf_files.append(doc_id)
                else:
                    pdf_files.append(f"{doc_id}.pdf")
            
            # Track all generated embeddings
            page_embeddings = {}
            document_page_map = {}
            
            # Process each PDF
            for pdf_file in tqdm(pdf_files, desc="Processing PDFs for visual index"):
                doc_id = os.path.splitext(pdf_file)[0]
                pdf_path = os.path.join(pdf_dir, pdf_file)
                
                # Skip if file doesn't exist
                if not os.path.exists(pdf_path):
                    logger.warning(f"PDF file not found: {pdf_path}")
                    continue
                
                # Convert PDF to images
                try:
                    pages = convert_from_path(pdf_path)
                except Exception as e:
                    logger.error(f"Error converting PDF {pdf_file} to images: {str(e)}")
                    traceback.print_exc()
                    continue
                
                # Process each page
                for page_idx, page_img in enumerate(pages):
                    page_id = f"{doc_id}_{page_idx}"
                    document_page_map[page_id] = {"doc_id": doc_id, "page_idx": page_idx}
                    
                    # Process and embed the image
                    try:
                        if self.vision_retriever in ["colpali", "colqwen"]:
                            # Process the image into model-compatible input
                            processed_image = self.vision_processor.process_images([page_img])
                            processed_image = {k: v.to(self.vision_model.device) for k, v in processed_image.items()}
                            
                            # Generate embedding
                            with torch.no_grad():
                                embedding = self.vision_model(**processed_image)
                            
                            # Save embedding to file
                            embedding_file = os.path.join(output_dir, f"{page_id}.pt")
                            torch.save(embedding.cpu(), embedding_file)
                            
                            # Save embedding for multi-vector scoring
                            page_embeddings[page_id] = embedding.cpu()
                    except Exception as e:
                        logger.error(f"Error processing page {page_idx} of PDF {pdf_file}: {str(e)}")
                        traceback.print_exc()
                        continue
            
            # Generate query embeddings
            query_embeddings = {}
            for _, row in tqdm(self.df.iterrows(), desc="Processing queries for visual index"):
                q_id = row['q_id']
                question = row['question']
                
                try:
                    # Process the question
                    processed_query = self.vision_processor.process_queries([question])
                    processed_query = {k: v.to(self.vision_model.device) for k, v in processed_query.items()}
                    
                    # Generate embedding
                    with torch.no_grad():
                        embedding = self.vision_model(**processed_query)
                    
                    query_embeddings[q_id] = embedding.cpu()
                    
                    # Save query embedding for future use
                    query_embedding_file = os.path.join(output_dir, f"query_{q_id}.pt")
                    torch.save(embedding.cpu(), query_embedding_file)
                except Exception as e:
                    logger.error(f"Error generating embedding for query {q_id}: {str(e)}")
                    traceback.print_exc()
            
            # Use multi-vector scoring to rank documents for each query
            results = []
            for q_id, query_emb in tqdm(query_embeddings.items(), desc="Ranking documents for queries"):
                try:
                    # Extract document information
                    document_info = None
                    for _, row in self.df.iterrows():
                        if row['q_id'] == q_id:
                            document_info = row
                            break
                    
                    if document_info is None:
                        continue
                    
                    question = document_info['question']
                    
                    # Get relevant documents for this query based on the dataset
                    relevant_docs = []
                    if 'documents' in document_info:
                        try:
                            docs = eval(document_info['documents'])
                            relevant_docs = [doc.split(".pdf")[0] for doc in docs]
                        except:
                            # If documents field is not valid, use all documents
                            traceback.print_exc()
                            relevant_docs = [os.path.splitext(f)[0] for f in pdf_files]
                    # elif 'doc_path' in document_info:
                    #     doc_path = document_info['doc_path']
                    #     if isinstance(doc_path, str) and doc_path.strip():
                    #         doc_id = os.path.basename(doc_path).split('.')[0]
                    #         relevant_docs.append(doc_id)
                    
                    if not relevant_docs:
                        relevant_docs = [os.path.splitext(f)[0] for f in pdf_files]
                    
                    # Filter page embeddings to only include relevant documents
                    relevant_page_embeddings = {}
                    for page_id, embedding in page_embeddings.items():
                        doc_id = page_id.rsplit('_', 1)[0]
                        if doc_id in relevant_docs:
                            relevant_page_embeddings[page_id] = embedding
                    
                    # Prepare for multi-vector scoring
                    qs = query_emb  # Query in batch format
                    ds = torch.cat([emb for emb in relevant_page_embeddings.values()], dim=0)
                    
                    print(qs[0].shape)
                    print(ds[0].shape)
                    print(ds[1].shape)
                    print(len(ds))


                    if len(ds) == 0:
                        logger.warning(f"No relevant document embeddings found for query {q_id}")
                        continue
                    
                    # Run the multi-vector scoring
                    scores = self.vision_processor.score_multi_vector(qs, ds)
                    scores = scores.flatten().numpy()
                    
                    # Get indices of scores in descending order
                    top_indices = np.argsort(-scores)
                    
                    # Map indices to document IDs
                    ranked_docs = np.array(list(relevant_page_embeddings.keys()))[top_indices]
                    
                    # Store results for each ranked document
                    for doc_id, score in zip(ranked_docs, scores[top_indices]):
                        results.append({
                            'q_id': q_id,
                            'document_id': doc_id,
                            'score': float(score),
                            'question': question
                        })
                
                except Exception as e:
                    logger.error(f"Error ranking documents for query {q_id}: {str(e)}")
                    traceback.print_exc()
            
            # Save results to CSV
            with open(self.vision_retrieval_file, 'w', newline='') as csvfile:
                fieldnames = ['q_id', 'document_id', 'score', 'question']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            logger.info(f"Visual index saved to {self.vision_retrieval_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error building visual index: {str(e)}")
            traceback.print_exc()
            return False

    def build_text_index(self):
        """
        Build text index for all documents in the dataset.
        
        Returns:
            bool: Success status
        """
        logger.info(f"Building text index using {self.text_retriever}")
        
        try:
            # Cache documents if not already done
            if not self.document_cache:
                self.cache_documents()
            
            # Prepare all documents
            all_chunks = []
            chunk_to_doc_mapping = []
            
            for doc_id, pages in tqdm(self.document_cache.items(), desc="Processing documents for text index"):
                all_text = "\n".join(pages)
                chunks = self.split_text(all_text)
                
                for chunk in chunks:
                    all_chunks.append(chunk)
                    arxiv_id, page_num = self.identify_document_and_page(chunk)
                    chunk_to_doc_mapping.append({
                        'chunk': chunk,
                        'chunk_pdf_name': arxiv_id if arxiv_id else doc_id,
                        'pdf_page_number': page_num if page_num is not None else 0
                    })
            
            # Initialize retriever based on selected method
            if self.text_retriever == "bm25":
                # BM25 indexing
                bm25_model = BM25Okapi([chunk.split() for chunk in all_chunks])
                
                # Process each query
                results = []
                for _, row in tqdm(self.df.iterrows(), desc="Processing queries for BM25"):
                    q_id = row['q_id']
                    question = row['question']
                    
                    # Get BM25 scores
                    try:
                        scores = bm25_model.get_scores(question.split())
                        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k*2]
                        
                        for rank, idx in enumerate(top_indices):
                            chunk_info = chunk_to_doc_mapping[idx]
                            results.append({
                                'q_id': q_id,
                                'question': question,
                                'chunk': all_chunks[idx],
                                'chunk_pdf_name': chunk_info['chunk_pdf_name'],
                                'pdf_page_number': chunk_info['pdf_page_number'],
                                'rank': rank + 1,
                                'score': scores[idx]
                            })
                    except Exception as e:
                        logger.error(f"Error processing query {q_id} with BM25: {str(e)}")
                        traceback.print_exc()
                
            elif self.text_retriever in ["minilm", "mpnet", "bge"]:
                # Create Chroma collection with sentence transformer embeddings
                chroma_client = chromadb.Client()
                collection_name = f"st_col_{uuid.uuid4().hex[:8]}"
                collection = chroma_client.create_collection(
                    collection_name,
                    embedding_function=self.st_embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                
                # Add documents to collection
                collection.add(
                    documents=all_chunks,
                    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
                )
                
                # Process each query
                results = []
                for _, row in tqdm(self.df.iterrows(), desc=f"Processing queries for {self.text_retriever.upper()}"):
                    q_id = row['q_id']
                    question = row['question']
                    
                    # Get nearest chunks
                    try:
                        query_results = collection.query(
                            query_texts=[question],
                            n_results=self.top_k*2
                        )
                        
                        for rank, (chunk_idx, score) in enumerate(zip(
                            [int(id.split('_')[1]) for id in query_results['ids'][0]],
                            query_results['distances'][0]
                        )):
                            chunk_info = chunk_to_doc_mapping[chunk_idx]
                            results.append({
                                'q_id': q_id,
                                'question': question,
                                'chunk': all_chunks[chunk_idx],
                                'chunk_pdf_name': chunk_info['chunk_pdf_name'],
                                'pdf_page_number': chunk_info['pdf_page_number'],
                                'rank': rank + 1,
                                'score': 1.0 - score  # Convert distance to similarity
                            })
                    except Exception as e:
                        logger.error(f"Error processing query {q_id} with {self.text_retriever}: {str(e)}")
                        traceback.print_exc()
            
            # Save results to CSV
            with open(self.text_retrieval_file, 'w', newline='') as csvfile:
                fieldnames = ['q_id', 'question', 'chunk', 'chunk_pdf_name', 'pdf_page_number', 'rank', 'score']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            logger.info(f"Text index saved to {self.text_retrieval_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error building text index: {str(e)}")
            traceback.print_exc()
            return False

    def retrieve_visual_contexts(self, query_id):
        """
        Retrieve visual contexts using the specified visual retriever.
        
        Args:
            query_id (str): The query ID
            
        Returns:
            list: Top-k visual contexts (images)
        """
        try:
            # Check if we need to build the index
            if not os.path.exists(self.vision_retrieval_file) or self.force_reindex:
                logger.info(f"Visual index not found or force reindex is enabled. Building index...")
                if not self.build_visual_index():
                    return []
            
            # Load the retrieval results
            df_retrieval = pd.read_csv(self.vision_retrieval_file)
            
            # Filter for the current query
            query_rows = df_retrieval[df_retrieval['q_id'] == query_id]
            if len(query_rows) == 0:
                logger.warning(f"No visual contexts found for query {query_id}")
                return []
            
            # Get top-k visual contexts
            top_k_rows = query_rows.nlargest(self.top_k, 'score')
            
            # Load the images from PDFs
            pages = []
            pdf_dir = os.path.join(self.data_dir, "docs")
            
            for _, row in top_k_rows.iterrows():
                try:
                    document_id = row['document_id']
                    # Extract base document ID and page number
                    base_doc_id, page_number = document_id.rsplit('_', 1)
                    page_number = int(page_number)
                    
                    # Find the PDF file
                    pdf_path = os.path.join(pdf_dir, f"{base_doc_id}.pdf")
                    if not os.path.exists(pdf_path):
                        logger.warning(f"PDF file not found: {pdf_path}")
                        continue
                    
                    # Convert PDF page to image
                    pdf_images = convert_from_path(pdf_path)
                    if page_number >= len(pdf_images):
                        logger.warning(f"Page {page_number} out of range for {pdf_path}")
                        continue
                    
                    image = pdf_images[page_number]
                    pages.append({
                        'image': image,
                        'document_id': document_id,
                        'page_number': page_number
                    })
                except Exception as e:
                    logger.error(f"Error loading PDF page for {row['document_id']}: {str(e)}")
                    traceback.print_exc()
            
            logger.info(f"Retrieved {len(pages)} visual contexts for query {query_id}")
            return pages
        
        except Exception as e:
            logger.error(f"Error retrieving visual contexts for query {query_id}: {str(e)}")
            traceback.print_exc()
            return []
            
    def retrieve_textual_contexts(self, query_id):
        """
        Retrieve textual contexts using the specified text retriever.
        
        Args:
            query_id (str): The query ID
            
        Returns:
            list: Top-k textual contexts

        """

        
        try:            
            if not os.path.exists(self.text_retrieval_file) or self.force_reindex:
                logger.info(f"Textual index not found or force reindex is enabled. Building index...")
                if not self.build_text_index():
                    return []

            self.build_text_index()

            retrieval_path = self.text_retrieval_file

            df_retrieval = pd.read_csv(retrieval_path)
            
            # Filter for the current query
            query_rows = df_retrieval[df_retrieval['q_id'] == query_id]
            if len(query_rows) == 0:
                logger.warning(f"No textual contexts found for query {query_id}")
                return []
            
            # Get top-k textual contexts
            top_k_rows = query_rows.sort_values(by='rank', ascending=True).head(self.top_k)
            
            # Extract the contexts
            contexts = []
            for _, row in top_k_rows.iterrows():
                contexts.append({
                    'chunk': row['chunk'],
                    'chunk_pdf_name': row['chunk_pdf_name'] if 'chunk_pdf_name' in row else row.get('document_id', 'unknown'),
                    'pdf_page_number': row['pdf_page_number'] if 'pdf_page_number' in row else row.get('page_number', 0)
                })
            
            logger.info(f"Retrieved {len(contexts)} textual contexts for query {query_id}")
            return contexts
        
        except Exception as e:
            logger.error(f"Error retrieving textual contexts for query {query_id}: {str(e)}")
            traceback.print_exc()
            return []

    def encode_image(self, pil_image):
        """Encode a PIL image to base64 string."""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")    
        return img_str

    def generate_visual_response(self, query, visual_contexts):

        """
        Generate a response based on visual contexts.
        
        Args:
            query (str): The user's question
            visual_contexts (list): List of visual contexts (images)
            
        Returns:
            dict: Generated response
        """
        try:
            # Extract just the images
            images = [ctx['image'] for ctx in visual_contexts]
            
            # Create prompt
            prompt_template = f"""
            You are tasked with answering a question based on the relevant pages of a PDF document. Provide your response in the following format:
            ## Evidence:

            ## Chain of Thought:

            ## Answer:

            ___
            Instructions:

            1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided pages and populate them in the "Evidence" section. For each element, include the type, content, and a brief explanation of its relevance.

            2. Chain of Thought: In the "Chain of Thought" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. You should perform computations if you need to to get to the answer. 

            3. Answer: {self.qa_prompt}
            ___
            Question: {query}
            """
            
            if self.llm_model == "gpt4":
                base64_images = [self.encode_image(img) for img in images]
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            } for base64_image in base64_images
                        ] + [
                            {"type": "text", "text": prompt_template}
                        ]
                    }
                ]

                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=messages,
                    max_tokens=3000,
                    temperature=0.7
                )
                
                return response.choices[0].message.content
                
            elif self.llm_model == "gemini":
                multimodal_prompt = [prompt_template] + images
                response = self.llm.generate_content(multimodal_prompt)
                return response.text
                
            elif self.llm_model == "qwen":
                messages = [
                    {"role": "user", "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt_template}]}
                ]
                
                text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = self.process_vision_info(messages)
                inputs = self.qwen_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
                
                return output_text[0]
        
        except Exception as e:
            logger.error(f"Error generating visual response: {str(e)}")
            traceback.print_exc()
            return "Error generating response from visual contexts."
    
    def generate_textual_response(self, query, textual_contexts):
        """
        Generate a response based on textual contexts.
        
        Args:
            query (str): The user's question
            textual_contexts (list): List of textual contexts
            
        Returns:
            dict: Generated response
        """
        try:
            # Extract the text chunks
            contexts = [ctx['chunk'] for ctx in textual_contexts]
            contexts_str = "\n- ".join(contexts)
            
            # Create prompt
            prompt_template = f"""
            You are tasked with answering a question based on the relevant chunks of a PDF document. Provide your response in the following format:
            ## Evidence:

            ## Chain of Thought:

            ## Answer:

            ___
            Instructions:

            1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided chunks and populate them in the "Evidence" section. For each element, include the type, content, and a brief explanation of its relevance.

            2. Chain of Thought: In the "Chain of Thought" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. You should perform computations if you need to to get to the answer. 

            3. Answer: {self.qa_prompt}
            ___
            Question: {query}
            ___
            Context: {contexts_str}
            
            """
            
            if self.llm_model == "gpt4":
                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=[
                        {"role": "user", "content": prompt_template}
                    ],
                    max_tokens=3000,
                    temperature=0.7
                )
                return response.choices[0].message.content
                
            elif self.llm_model == "gemini":
                response = self.llm.generate_content(prompt_template)
                return response.text
                
            elif self.llm_model == "qwen":
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt_template}]}
                ]
                
                text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.qwen_processor(text=[text], padding=True, return_tensors="pt").to("cuda")

                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
                
                return output_text[0]
        
        except Exception as e:
            logger.error(f"Error generating textual response: {str(e)}")
            return "Error generating response from textual contexts."

    def extract_sections(self, text):
        """
        Extract sections from the generated text.
        
        Args:
            text (str): The generated text
            
        Returns:
            dict: Extracted sections
        """
        sections = {}
        headings = ["Evidence", "Chain of Thought", "Answer"]
        
        for i in range(len(headings)):
            heading = headings[i]
            next_heading = headings[i + 1] if i + 1 < len(headings) else None
            
            if next_heading:
                pattern = rf"## {heading}:(.*?)(?=## {next_heading}:)"
            else:
                pattern = rf"## {heading}:(.*)"
            
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[heading] = match.group(1).strip()
            else:
                sections[heading] = ""
        
        return sections

    def combine_responses(self, query, visual_response, textual_response, answer):
        """
        Combine visual and textual responses to generate a final answer.
        
        Args:
            query (str): The user's question
            visual_response (dict): Response from visual contexts
            textual_response (dict): Response from textual contexts
            answer (str): Ground truth answer if available
            
        Returns:
            dict: Combined response
        """
        try:
            prompt = f"""
            Analyze the following two responses to the question: "{query}"

            Response 1:
            Evidence: {visual_response.get('Evidence', "Evidence not available")}
            Chain of Thought: {visual_response.get('Chain of Thought', "CoT not available")}
            Final Answer: {visual_response['Answer']}

            Response 2:
            Evidence: {textual_response.get('Evidence', "Evidence not available")}
            Chain of Thought: {textual_response.get('Chain of Thought', "CoT not available")}
            Final Answer: {textual_response['Answer']}

            Response 1 is based on a visual q/a pipeline, and Response 2 is based on a textual q/a pipeline. 
            - In general, given both response 1 and response 2 have logical chains of thoughts, and decision boils down to evidence, you should place higher degree of trust on evidence reported in Response 1.
            - If one of the responses has declined giving a clear answer, please weigh the other answer more unless there is reasonable thought to not answer, and both thoughts are inconsistent.
            - Language of the answer should be short and direct, usually answerable in a single sentence, or phrase. You should directly give the specific response to an answer.

            Consider both chains of thought and final answers. Provide your analysis in the following format:

            ## Analysis:
            [Your detailed analysis here, evaluating the consistency of both the chains of thoughts, with respect to each other, the question and their respective answers, as well as validity of the evidence.]

            ## Conclusion:
            [Your conclusion on which answer is more likely to be correct, or if a synthesis of both is needed]

            ## Final Answer:
            [Answer the question "{query}", based on your analysis of the two candidates so far. Please ensure that answers are short and concise, similar in language to the provided answers.]
            """
            
            if self.llm_model == "gpt4":
                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                return self.parse_combined_output(response.choices[0].message.content)
                
            elif self.llm_model == "gemini":
                response = self.llm.generate_content(prompt)
                return self.parse_combined_output(response.text)
                
            elif self.llm_model == "qwen":
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
                
                text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.qwen_processor(text=[text], padding=True, return_tensors="pt").to("cuda")

                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1000)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
                
                return self.parse_combined_output(output_text[0])
        
        except Exception as e:
            logger.error(f"Error combining responses: {str(e)}")
            return {
                "Analysis": "Error occurred during analysis.",
                "Conclusion": "Error occurred during conclusion.",
                "Final Answer": "Error occurred during combination of responses."
            }

    def parse_combined_output(self, output):
        """
        Parse the output of the combination step.
        
        Args:
            output (str): The combined output text
            
        Returns:
            dict: Parsed sections
        """
        sections = {'Analysis': '', 'Conclusion': '', 'Final Answer': ''}
        current_section = None

        for line in output.split('\n'):
            if line.startswith('## '):
                current_section = line[3:].strip(':')
            elif current_section and current_section in sections:
                sections[current_section] += line + '\n'

        # Clean up the sections
        for key in sections:
            sections[key] = sections[key].strip()

        return sections

    def process_query(self, query_id):
        """
        Process a single query through the complete VisDoMRAG pipeline.
        
        Args:
            query_id (str): The query ID
            
        Returns:
            bool: Success status
        """
        try:
            # Get query information
            query_row = self.df[self.df['q_id'] == query_id].iloc[0]
            question = query_row['question']
            
            try:
                # Try to parse the answer field as a list/dict if it's in that format
                answer = eval(query_row['answer'])
            except:
                # If parsing fails, use as-is
                answer = query_row['answer']
            
            # Define file paths for outputs
            visual_file = f"{self.output_dir}/{self.llm_model}_vision/response_{str(query_id).replace('/','$')}.json"
            textual_file = f"{self.output_dir}/{self.llm_model}_text/response_{str(query_id).replace('/','$')}.json"
            combined_file = f"{self.output_dir}/{self.llm_model}_visdmrag/response_{str(query_id).replace('/','$')}.json"
            
            # Skip if the combined file already exists
            if os.path.exists(combined_file):
                logger.info(f"Combined file already exists for query {query_id}")
                return True
            
            # Process visual contexts if needed
            visual_response_dict = None
            if not os.path.exists(visual_file):
                logger.info(f"Generating visual response for query {query_id}")
                visual_contexts = self.retrieve_visual_contexts(query_id)
                
                if visual_contexts:
                    visual_response = self.generate_visual_response(question, visual_contexts)
                    visual_response_dict = self.extract_sections(visual_response)
                    
                    # Add metadata
                    visual_response_dict.update({
                        "question": question,
                        "document": [ctx['document_id'] for ctx in visual_contexts],
                        "gt_answer": answer,
                        "pages": [ctx['page_number'] for ctx in visual_contexts]
                    })
                    
                    # Save visual response
                    with open(visual_file, 'w') as file:
                        json.dump(visual_response_dict, file, indent=4)
                    
                    # Memory cleanup
                    del visual_contexts
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                # Load existing visual response
                with open(visual_file, 'r') as file:
                    visual_response_dict = json.load(file)
            
            # Process textual contexts if needed
            textual_response_dict = None
            if not os.path.exists(textual_file):
                logger.info(f"Generating textual response for query {query_id}")
                textual_contexts = self.retrieve_textual_contexts(query_id)
                
                if textual_contexts:
                    textual_response = self.generate_textual_response(question, textual_contexts)
                    textual_response_dict = self.extract_sections(textual_response)
                    
                    # Add metadata
                    textual_response_dict.update({
                        "question": question,
                        "document": [ctx['chunk_pdf_name'] for ctx in textual_contexts],
                        "gt_answer": answer,
                        "pages": [ctx['pdf_page_number'] for ctx in textual_contexts],
                        "chunks": "\n".join([ctx['chunk'] for ctx in textual_contexts])
                    })
                    
                    # Save textual response
                    with open(textual_file, 'w') as file:
                        json.dump(textual_response_dict, file, indent=4)
                    
                    # Memory cleanup
                    del textual_contexts
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                # Load existing textual response
                with open(textual_file, 'r') as file:
                    textual_response_dict = json.load(file)
            
            # Skip if either response is missing
            if not visual_response_dict or not textual_response_dict:
                logger.warning(f"Missing responses for query {query_id}")
                return False
            
            # Combine responses
            logger.info(f"Combining responses for query {query_id}")
            combined_sections = self.combine_responses(
                question, 
                visual_response_dict, 
                textual_response_dict,
                answer
            )
            
            # Create combined response
            combined_response = {
                "question": question,
                "answer": combined_sections.get("Final Answer", ""),
                "gt_answer": answer,
                "analysis": combined_sections.get("Analysis", ""),
                "conclusion": combined_sections.get("Conclusion", ""),
                "response1": visual_response_dict,
                "response2": textual_response_dict
            }
            
            # Save combined response
            with open(combined_file, 'w') as file:
                json.dump(combined_response, file, indent=4)
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            return False

    def run(self):
        """Run the VisDoMRAG pipeline on all queries in the dataset."""
        logger.info("Starting VisDoMRAG pipeline")
        
        # Process each query
        for query_id in tqdm(self.df['q_id'].unique()):
            try:
                logger.info(f"Processing query {query_id}")
                success = self.process_query(query_id)
                
                if success:
                    logger.info(f"Successfully processed query {query_id}")
                else:
                    logger.warning(f"Failed to process query {query_id}")
                
                # Pause to avoid hitting API rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing query {query_id}: {str(e)}")
        
        logger.info("VisDoMRAG pipeline completed")

