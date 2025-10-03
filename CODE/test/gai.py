import asyncio
import aiohttp
import json
import logging
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import time
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 not installed. Install with: pip install PyPDF2")
    exit(1)

try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
except ImportError:
    print("ReportLab not installed. Install with: pip install reportlab")
    exit(1)

try:
    from charset_normalizer import detect
except ImportError:
    print("charset-normalizer not installed. Install with: pip install charset-normalizer")
    exit(1)

try:
    import structlog
except ImportError:
    print("structlog not installed. Install with: pip install structlog")
    exit(1)

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    print("tqdm not installed. Install with: pip install tqdm")
    exit(1)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@dataclass
class ChunkData:
    content: str
    chunk_num: int
    elements: List[Dict[str, Any]]
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()[:8]

@dataclass
class TranslationConfig:
    api_url: str = "http://127.0.0.1:1234/v1/chat/completions"
    model: str = "gpt-3.5-turbo"
    max_chars: int = 2000
    max_tokens: int = 4000
    temperature: float = 0.1
    max_retries: int = 3
    retry_delay: float = 2.0
    request_timeout: int = 90
    max_concurrent_requests: int = 3
    rate_limit_delay: float = 1.0

    def __post_init__(self):
        if not self.api_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid API URL: {self.api_url}")
        if self.max_chars < 100:
            raise ValueError(f"max_chars must be >= 100, got {self.max_chars}")
        if self.max_tokens < 100:
            raise ValueError(f"max_tokens must be >= 100, got {self.max_tokens}")
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")
        if self.retry_delay < 0:
            raise ValueError(f"retry_delay must be >= 0, got {self.retry_delay}")
        if self.request_timeout < 10:
            raise ValueError(f"request_timeout must be >= 10, got {self.request_timeout}")
        if self.max_concurrent_requests < 1:
            raise ValueError(f"max_concurrent_requests must be >= 1, got {self.max_concurrent_requests}")

    @classmethod
    def from_env_or_input(cls) -> 'TranslationConfig':
        api_url = os.getenv("API_URL", input("Enter API URL (press Enter for default): ").strip() or cls.api_url)
        config = cls(api_url=api_url)
        
        if input("Configure advanced settings? (y/n): ").lower().strip() == 'y':
            try:
                config.max_concurrent_requests = int(
                    os.getenv("MAX_CONCURRENT", 
                              input(f"Max concurrent requests (default {config.max_concurrent_requests}): ").strip() or config.max_concurrent_requests)
                )
                config.max_chars = int(
                    os.getenv("CHUNK_SIZE", 
                              input(f"Chunk size (default {config.max_chars}): ").strip() or config.max_chars)
                )
            except ValueError:
                logger.warning("Invalid input for advanced settings; using defaults.")
        return config

class OptimizedAPITester:
    @staticmethod
    async def test_api_connection_async(api_url: str, session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
        try:
            test_payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }

            async with session.post(api_url, json=test_payload, timeout=15) as response:  # type: ignore
                if response.status != 200:
                    text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {text[:200]}"
                    }
                
                result = await response.json()
                
                if 'choices' not in result or not result['choices']:
                    return {
                        "success": False,
                        "error": "Invalid API response structure"
                    }
                
                return {
                    "success": True,
                    "response": result,
                    "test_message": result['choices'][0]['message']['content']
                }
                
        except aiohttp.ClientConnectionError as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

class OptimizedDocumentStructure:
    def __init__(self, text: str):
        self.original_text = text
        self._lines = text.split('\n')
        self.structure = self._analyze_structure_optimized()
    
    @lru_cache(maxsize=1000)
    def _is_heading(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > 100:
            return False
        
        heading_patterns = [
            re.compile(r'^[A-Z\s]{3,}$'),
            re.compile(r'^\d+\.\s+[A-Z]'),
            re.compile(r'^Chapter\s+\d+', re.IGNORECASE),
            re.compile(r'^Section\s+\d+', re.IGNORECASE),
            re.compile(r'^[IVX]+\.\s+'),
        ]
        
        if any(pattern.match(stripped) for pattern in heading_patterns):
            return True
        
        return (len(stripped) < 50 and 
                not stripped.endswith(('.', '!', '?', ';', ':')) and
                len(stripped.split()) <= 8 and
                any(word[0].isupper() for word in stripped.split() if word))
    
    def _analyze_structure_optimized(self) -> Dict[str, Any]:
        structure = {
            'paragraphs': [],
            'headings': [],
            'total_lines': len(self._lines)
        }
        
        current_paragraph = []
        paragraph_start = 0
        
        for i, line in enumerate(self._lines):
            stripped = line.strip()
            
            if not stripped:  # Empty line
                if current_paragraph:
                    structure['paragraphs'].append({
                        'content': '\n'.join(current_paragraph),
                        'start_line': paragraph_start,
                        'end_line': i - 1,
                        'length': sum(len(l) for l in current_paragraph)
                    })
                    current_paragraph = []
                paragraph_start = i + 1
                
            elif self._is_heading(line):
                if current_paragraph:
                    structure['paragraphs'].append({
                        'content': '\n'.join(current_paragraph),
                        'start_line': paragraph_start,
                        'end_line': i - 1,
                        'length': sum(len(l) for l in current_paragraph)
                    })
                    current_paragraph = []
                
                structure['headings'].append({
                    'content': stripped,
                    'line': i,
                    'level': self._get_heading_level(line)
                })
                paragraph_start = i + 1
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            structure['paragraphs'].append({
                'content': '\n'.join(current_paragraph),
                'start_line': paragraph_start,
                'end_line': len(self._lines) - 1,
                'length': sum(len(l) for l in current_paragraph)
            })
        
        return structure
    
    def _get_heading_level(self, line: str) -> int:
        stripped = line.strip()
        if re.match(r'^[A-Z\s]{3,}$', stripped):
            return 1
        elif re.match(r'^\d+\.\s+', stripped):
            return 2
        else:
            return 3

class OptimizedPDFTranslator:
    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self._session = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        await self._test_api_setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def _test_api_setup(self):
        test_result = await OptimizedAPITester.test_api_connection_async(
            self.config.api_url,
            self._session
        )
        
        if test_result["success"]:
            logger.info("API connection successful")
        else:
            logger.error("API connection failed", error=test_result['error'])
            raise Exception(f"API setup failed: {test_result['error']}")
    
    @lru_cache(maxsize=100)
    def _detect_encoding(self, file_path: str) -> Optional[str]:
        """Detect file encoding using charset-normalizer."""
        try:
            with open(file_path, 'rb') as f:
                result: Dict[str, Any] = detect(f.read(4096))  # Read 4KB for detection #  type: ignore
                if result['encoding'] and result.get('confidence', 0.0) > 0.8:
                    logger.debug("Encoding detected", file=file_path, encoding=result['encoding'], confidence=result['confidence'])
                    return result['encoding']
                logger.debug("Encoding not reliable", file=file_path, encoding=result.get('encoding'), confidence=result.get('confidence', 0.0))
                return None
        except Exception as e:
            logger.warning("Encoding detection failed", file=file_path, error=str(e))
            return None

    def extract_text_from_file(self, file_path: str) -> str:
        encodings = [self._detect_encoding(file_path), 'utf-8', 'windows-1258', 'latin-1', 'iso-8859-1']
        encodings = [e for e in encodings if e]  # Remove None

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, buffering=8192) as f:
                    content = f.read()
                logger.info("Text extracted", file=file_path, encoding=encoding)
                if not content.strip():
                    raise ValueError("Text file is empty")
                return content.strip()
            except (UnicodeDecodeError, UnicodeError) as e:
                logger.debug("Encoding failed", encoding=encoding, error=str(e))
                continue

        logger.error("Failed to decode file", file=file_path, encodings_tried=encodings)
        raise ValueError(
            f"Could not decode text file {file_path} with any supported encoding. "
            f"Tried: {', '.join(encodings)}. "
            "Try converting the file to UTF-8."
        )
    
    def _extract_from_pdf_optimized(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            max_workers = min(os.cpu_count() * 2, num_pages) if num_pages > 10 else 1 # type: ignore
            text_parts = ["" for _ in range(num_pages)]

            def process_page(page_num: int, page) -> Tuple[int, str]:
                try:
                    text = page.extract_text()
                    return page_num, f"--- Page {page_num + 1} ---\n\n{text}" if text else ""
                except Exception as e:
                    logger.warning("Page extraction error", page_num=page_num+1, error=str(e))
                    return page_num, ""

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_page, i, page)
                    for i, page in enumerate(reader.pages)
                ]
                for future in futures:
                    page_num, page_text = future.result()
                    if page_text:
                        text_parts[page_num] = page_text

            full_text = '\n\n'.join(t for t in text_parts if t)
            if not full_text.strip():
                raise ValueError("No text extracted from PDF")

            logger.info("PDF extracted", chars=len(full_text), pages=num_pages)
            return full_text.strip()

        except Exception as e:
            logger.error("PDF extraction failed", error=str(e))
            raise
    
    def intelligent_chunk_optimized(self, doc_structure: OptimizedDocumentStructure) -> List[ChunkData]:
        """Optimized chunking with streaming to reduce memory usage."""
        def generate_chunks():
            current_chunk = []
            current_length = 0
            current_elements = []
            chunk_num = 1  # Initialize chunk counter

            all_elements = sorted(
                [(h['line'], 'heading', h) for h in doc_structure.structure['headings']] +
                [(p['start_line'], 'paragraph', p) for p in doc_structure.structure['paragraphs']],
                key=lambda x: x[0]
            )

            for line_num, element_type, element in all_elements:
                content = element['content']
                content_length = len(content)

                if current_length + content_length > self.config.max_chars and current_chunk:
                    yield ChunkData(
                        content='\n\n'.join(current_chunk),
                        chunk_num=chunk_num,
                        elements=current_elements
                    )
                    current_chunk = []
                    current_length = 0
                    current_elements = []
                    chunk_num += 1  # Increment chunk counter

                current_chunk.append(content)
                current_length += content_length + 2
                current_elements.append({
                    'type': element_type,
                    'content': content,
                    'line': line_num
                })

            if current_chunk:
                yield ChunkData(
                    content='\n\n'.join(current_chunk),
                    chunk_num=chunk_num,
                    elements=current_elements
                )

        chunks = list(generate_chunks())
        logger.info("Chunks created", count=len(chunks))
        return chunks
    
    async def process_chunk_async(self, chunk_data: ChunkData, operation: str = "translate", 
                                user_prompt: Optional[str] = None) -> Optional[str]:
        if len(chunk_data.content) < 500:
            return (await self._process_batched_chunks([chunk_data], operation, user_prompt))[0]
        async with self._semaphore:
            return await self._process_single_chunk(chunk_data, operation, user_prompt)
    
    async def _process_batched_chunks(self, chunk_data_list: List[ChunkData], operation: str,
                                    user_prompt: Optional[str]) -> List[Optional[str]]:
        if self._session is None:
            logger.error("Session not initialized for batch processing")
            return [None] * len(chunk_data_list)
        
        if not chunk_data_list:
            logger.error("Empty chunk list provided for batch processing")
            return []
        
        combined_content = "\n\n---CHUNK_SEPARATOR---\n\n".join(c.content for c in chunk_data_list)
        if not combined_content.strip():
            logger.error("Empty combined content for batch processing")
            return [None] * len(chunk_data_list)
        
        if operation == "translate":
            prompt = f"""Translate the following text from English to Vietnamese. 
                    Each chunk is separated by '---CHUNK_SEPARATOR---'. Return the translated chunks with the same separator.
                    Preserve ALL formatting, structure, headings, and paragraph breaks.

                    Text to translate:

                    {combined_content}"""
            system_msg = "You are a professional translator. Translate accurately while preserving ALL formatting."
        else:
            prompt = f"""Rewrite the following text. Each chunk is separated by '---CHUNK_SEPARATOR---'. 
                        Return the rewritten chunks with the same separator. Preserve ALL formatting.
                        Text to rewrite: {combined_content}"""
            system_msg = "You are a professional rewriter. Rewrite accurately while preserving ALL formatting."

        estimated_tokens = len(combined_content) // 4 + 100
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": min(estimated_tokens * 2, self.config.max_tokens),
            "temperature": self.config.temperature
        }

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    async with self._session.post(self.config.api_url, json=payload) as response:
                        if response.status != 200:
                            text = await response.text()
                            logger.error("Batch HTTP error", status=response.status, response_text=text[:200], attempt=attempt+1)
                            if attempt < self.config.max_retries - 1:
                                await asyncio.sleep(self.config.retry_delay)
                            continue
                        result = await response.json()
                        if not result.get('choices') or not result['choices'][0].get('message', {}).get('content'):
                            logger.error("Invalid batch response structure", response=result, attempt=attempt+1)
                            if attempt < self.config.max_retries - 1:
                                await asyncio.sleep(self.config.retry_delay)
                            continue
                        processed_text = result['choices'][0]['message']['content']
                        return processed_text.split("\n\n---CHUNK_SEPARATOR---\n\n")
                except aiohttp.ClientConnectionError as e:
                    logger.error("Batch connection error", error=str(e), error_type=type(e).__name__, attempt=attempt+1)
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                except aiohttp.ClientResponseError as e:
                    logger.error("Batch response error", error=str(e), status=e.status, error_type=type(e).__name__, attempt=attempt+1)
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                except asyncio.TimeoutError:
                    logger.error("Batch request timeout", attempt=attempt+1)
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                except Exception as e:
                    logger.error("Batch unexpected error", error=str(e), error_type=type(e).__name__, traceback="".join(traceback.format_tb(e.__traceback__)), attempt=attempt+1)
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
            logger.error("Batch processing failed after retries")
            return [None] * len(chunk_data_list)
    
    async def _process_single_chunk(self, chunk_data: ChunkData, operation: str, 
                                  user_prompt: Optional[str]) -> Optional[str]:
        logger.info("Processing chunk", chunk_num=chunk_data.chunk_num, chars=len(chunk_data.content))
        
        if not chunk_data.content.strip():
            logger.error("Empty chunk content", chunk_num=chunk_data.chunk_num)
            return None
        
        if operation == "translate":
            prompt = self._create_translation_prompt(chunk_data.content)
            system_msg = "You are a professional translator. Translate accurately while preserving ALL formatting, structure, headings, and paragraph breaks."
        else:
            prompt = self._create_rewrite_prompt(chunk_data.content, user_prompt)
            system_msg = "You are a professional rewriter. Rewrite accurately while preserving ALL formatting, structure, headings, and paragraph breaks."
        
        estimated_tokens = len(chunk_data.content) // 4 + 100
        max_tokens = min(estimated_tokens * 2, self.config.max_tokens)

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": self.config.temperature
        }
        
        if self._session is None:
            logger.error("Aiohttp session not initialized", chunk_num=chunk_data.chunk_num)
            return None
        
        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(self.config.api_url, json=payload) as response:
                    if response.status == 429:
                        retry_after = float(response.headers.get("Retry-After", self.config.retry_delay))
                        logger.warning("Rate limit hit", chunk_num=chunk_data.chunk_num, attempt=attempt+1, retry_after=retry_after)
                        await asyncio.sleep(retry_after * (2 ** attempt))
                        continue
                    if response.status != 200:
                        text = await response.text()
                        logger.error("HTTP error", chunk_num=chunk_data.chunk_num, status=response.status, response_text=text[:200], attempt=attempt+1)
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                        continue
                    result = await response.json()
                    if not result.get('choices') or not result['choices'][0].get('message', {}).get('content'):
                        logger.error("Invalid response structure", chunk_num=chunk_data.chunk_num, response=result, attempt=attempt+1)
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                        continue
                    processed_text = result['choices'][0]['message']['content']
                    logger.info("Chunk processed", chunk_num=chunk_data.chunk_num)
                    return processed_text
            except aiohttp.ClientConnectionError as e:
                logger.error("Connection error", chunk_num=chunk_data.chunk_num, error=str(e), error_type=type(e).__name__, attempt=attempt+1)
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
            except aiohttp.ClientResponseError as e:
                logger.error("Response error", chunk_num=chunk_data.chunk_num, error=str(e), status=e.status, error_type=type(e).__name__, attempt=attempt+1)
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
            except asyncio.TimeoutError:
                logger.error("Request timeout", chunk_num=chunk_data.chunk_num, attempt=attempt+1)
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
            except Exception as e:
                logger.error("Unexpected error", chunk_num=chunk_data.chunk_num, error=str(e), error_type=type(e).__name__, traceback="".join(traceback.format_tb(e.__traceback__)), attempt=attempt+1)
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
        
        logger.error("Chunk processing failed after retries", chunk_num=chunk_data.chunk_num)
        return None
    
    def _create_translation_prompt(self, content: str) -> str:
        return f"""Translate the following text from English to Vietnamese. 
                IMPORTANT: Preserve all formatting, structure, headings, and paragraph breaks exactly as they appear in the original.

                Text to translate:

                {content}"""
    
    def _create_rewrite_prompt(self, content: str, user_prompt: Optional[str]) -> str:
        if user_prompt:
            return f"""Find and rewrite the content that similar to "{user_prompt}". If not found, keep the current content unchanged.
                    IMPORTANT: Preserve all formatting, structure, headings, and paragraph breaks exactly as they appear in the original.

                    Text to rewrite: {content}"""
        else:
            return f"""Rewrite and improve the following text while maintaining its original meaning.
                    IMPORTANT: Preserve all formatting, structure, headings, and paragraph breaks exactly as they appear in the original.

                    Text to rewrite: {content}"""
    
    async def translate_file_async(self, input_file: str, output_pdf: Optional[str] = None, 
                                 output_txt: Optional[str] = None):
        await self._process_file_async(input_file, "translate", None, output_pdf, output_txt)
    
    async def rewrite_file_async(self, input_file: str, user_prompt: Optional[str] = None,
                               output_pdf: Optional[str] = None, output_txt: Optional[str] = None):
        await self._process_file_async(input_file, "rewrite", user_prompt, output_pdf, output_txt)
    
    async def _process_file_async(self, input_file: str, operation: str, user_prompt: Optional[str],
                                output_pdf: Optional[str], output_txt: Optional[str]):
        try:
            logger.info("Starting workflow", operation=operation, file=input_file)
            start_time = time.time()
            
            # Debug: Log available methods
            logger.debug("Available methods", methods=[m for m in dir(self) if callable(getattr(self, m))])
            
            # Check if process_chunk_async exists
            if not hasattr(self, 'process_chunk_async'):
                logger.error("process_chunk_async method missing", class_name=self.__class__.__name__)
                raise AttributeError("process_chunk_async method not found in OptimizedPDFTranslator")
            
            # Determine if input is PDF or text
            input_path = Path(input_file)
            if input_path.suffix.lower() not in {'.pdf', '.txt'}:
                raise ValueError(f"Unsupported file type: {input_path.suffix}. Only .pdf and .txt are supported.")
            
            if input_path.suffix.lower() == '.pdf':
                full_text = self._extract_from_pdf_optimized(input_file)
            else:
                full_text = self.extract_text_from_file(input_file)
            logger.info("Text extracted", length=len(full_text))
            
            doc_structure = OptimizedDocumentStructure(full_text)
            logger.info("Structure analyzed", paragraphs=len(doc_structure.structure['paragraphs']),
                       headings=len(doc_structure.structure['headings']))
            
            chunks = self.intelligent_chunk_optimized(doc_structure)
            
            if len(chunks) > 10:
                logger.warning("Large document", chunk_count=len(chunks))
                proceed = input("This may take a while. Continue? (y/n): ").lower().strip()
                if proceed != 'y':
                    logger.info("Operation cancelled")
                    return
            
            logger.info("Processing chunks", total=len(chunks))
            tasks = [
                self.process_chunk_async(chunk, operation, user_prompt)
                for chunk in chunks
            ]
            
            results = await tqdm_asyncio.gather(*tasks, desc="Processing chunks")
            
            processed_chunks = []
            failed_chunks = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception) or result is None:
                    logger.error("Chunk failed", chunk_num=i+1, error=str(result) if isinstance(result, Exception) else "None")
                    failed_chunks.append(i+1)
                    processed_chunks.append(f"[{operation.upper()} FAILED FOR CHUNK {i+1}]")
                else:
                    processed_chunks.append(result)
            
            full_result = "\n\n".join(processed_chunks)
            
            input_path = Path(input_file)
            suffix = "_vietnamese" if operation == "translate" else "_rewritten"
            if output_pdf is None:
                output_pdf = f"{input_path.stem}{suffix}.pdf"
            if output_txt is None:
                output_txt = f"{input_path.stem}{suffix}.txt"
            
            self.save_text_backup(full_result, output_txt)
            self.create_formatted_pdf_optimized(full_result, output_pdf, doc_structure)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info("Workflow completed", time=f"{processing_time:.2f}s",
                       output_pdf=output_pdf, output_txt=output_txt, failed_chunks=len(failed_chunks))
            
        except Exception as e:
            logger.error("Workflow failed", error=str(e))
            raise
    
    def save_text_backup(self, text: str, filename: str):
        try:
            with open(filename, 'w', encoding='utf-8', buffering=8192) as f:
                f.write(text)
            logger.info("Text backup saved", filename=filename)
        except Exception as e:
            logger.error("Failed to save text backup", error=str(e))
            raise
    
    def create_formatted_pdf_optimized(self, text: str, filename: str, 
                                     doc_structure: OptimizedDocumentStructure):
        try:
            pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))

            doc = SimpleDocTemplate(
                filename, 
                pagesize=letter,
                topMargin=1*inch, 
                bottomMargin=1*inch,
                leftMargin=1*inch, 
                rightMargin=1*inch
            )
            
            styles = getSampleStyleSheet()
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=12,
                alignment=TA_LEFT
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                fontName='Times-Roman'
            )
            
            story = []
            
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            
            for para_text in paragraphs:
                para_text = para_text.strip()
                
                if para_text.startswith('[') and 'FAILED' in para_text:
                    p = Paragraph(para_text, styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 12))
                    continue
                
                if self._is_heading_for_pdf(para_text):
                    p = Paragraph(para_text, heading_style)
                    story.append(p)
                else:
                    lines = para_text.split('\n')
                    for line in lines:
                        if line.strip():
                            p = Paragraph(line.strip(), normal_style)
                            story.append(p)
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            logger.info("Formatted PDF created", filename=filename)
            
        except Exception as e:
            logger.error("PDF creation failed", error=str(e))
            raise

    @lru_cache(maxsize=500)
    def _is_heading_for_pdf(self, text: str) -> bool:
        """
        Determine if a text line should be formatted as a heading in PDF.
        """
        text = text.strip()
        
        # Check for common heading patterns
        heading_patterns = [
            r'^#{1,6}\s+',  # Markdown headers (# ## ### etc.)
            r'^[A-Z][A-Z\s]{2,}:?\s*$',  # ALL CAPS headings
            r'^\d+\.\s+[A-Z]',  # Numbered sections (1. Introduction)
            r'^[IVX]+\.\s+[A-Z]',  # Roman numeral sections (I. Introduction)
            r'^Chapter\s+\d+',  # Chapter headings
            r'^Section\s+\d+',  # Section headings
            r'^Part\s+[IVX\d]+',  # Part headings
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Check if text is short and starts with capital letter (likely a heading)
        if len(text) < 80 and text[0].isupper() and not text.endswith('.'):
            # Additional checks to avoid false positives
            word_count = len(text.split())
            if 2 <= word_count <= 10:  # Reasonable heading length
                return True
        
        # Check if line ends with colon (often indicates a heading or section start)
        if text.endswith(':') and len(text.split()) <= 8:
            return True
            
        return False

    def _escape_html_chars(self, text: str) -> str:
        """
        Escape HTML/XML characters that might cause issues in ReportLab paragraphs.
        """
        html_escape_chars = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;'
        }
        
        for char, escape in html_escape_chars.items():
            text = text.replace(char, escape)
        
        return text

    def _handle_special_formatting(self, text: str, style: ParagraphStyle) -> Paragraph:
        """
        Handle special text formatting like bold, italic, etc.
        """
        # Escape HTML characters first
        text = self._escape_html_chars(text)
        
        # Handle markdown-style formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italic
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)  # Code
        
        return Paragraph(text, style)

# Convenience functions
async def translate_file_optimized(input_file: str, config: Optional[TranslationConfig] = None,
                                 output_pdf: Optional[str] = None, output_txt: Optional[str] = None):
    async with OptimizedPDFTranslator(config) as translator:
        await translator.translate_file_async(input_file, output_pdf, output_txt)

async def rewrite_file_optimized(input_file: str, user_prompt: Optional[str] = None,
                               config: Optional[TranslationConfig] = None,
                               output_pdf: Optional[str] = None, output_txt: Optional[str] = None):
    async with OptimizedPDFTranslator(config) as translator:
        await translator.rewrite_file_async(input_file, user_prompt, output_pdf, output_txt)

def check_dependencies():
    missing = []
    try:
        import PyPDF2
    except ImportError:
        missing.append("PyPDF2")
    try:
        import reportlab
    except ImportError:
        missing.append("reportlab")
    try:
        import charset_normalizer
    except ImportError:
        missing.append("charset-normalizer")
    try:
        import structlog # type: ignore
    except ImportError:
        missing.append("structlog")
    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        exit(1)

def main():
    check_dependencies()
    print("=== Optimized PDF/TXT Translator ===")
    print("Enhanced version with async processing and better performance")
    
    while True:
        input_file = input("\nEnter path to PDF/TXT file: ").strip()
        if not input_file:
            print("Please enter a file path.")
            continue
        if not Path(input_file).exists():
            print(f"File '{input_file}' not found. Please check the path.")
            continue
        break
    
    operation = input("Choose operation - (t)ranslate or (r)ewrite: ").lower().strip()
    if operation not in ['t', 'translate', 'r', 'rewrite']:
        operation = 't'
    
    user_prompt = None
    if operation in ['r', 'rewrite']:
        user_prompt = input("Enter rewrite prompt (or press Enter for general rewrite): ").strip() or None
    
    config = TranslationConfig.from_env_or_input()
    
    output_pdf = None
    output_txt = None
    if input("Use custom output names? (y/n): ").lower().strip() == 'y':
        output_pdf = input("PDF output filename (leave empty for auto): ").strip() or None
        output_txt = input("TXT output filename (leave empty for auto): ").strip() or None
    
    async def run_operation():
        try:
            print(f"\nInitializing translator with API: {config.api_url}")
            print(f"Max concurrent requests: {config.max_concurrent_requests}")
            async with OptimizedPDFTranslator(config) as translator:
                if operation in ['r', 'rewrite']:
                    await translator.rewrite_file_async(input_file, user_prompt, output_pdf, output_txt)
                    print("Rewrite completed successfully!")
                else:
                    await translator.translate_file_async(input_file, output_pdf, output_txt)
                    print("Translation completed successfully!")
        except Exception as e:
            logger.error("Operation failed", error=str(e))
            print(f"\nError: {e}")
            print("\nTroubleshooting:")
            print("1. Ensure your API server is running")
            print("2. Check the API URL is correct") 
            print("3. Verify file permissions")
            print("4. Check available disk space")
            print("5. Try reducing concurrent requests if getting rate limited")
    
    try:
        asyncio.run(run_operation())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error("Fatal error", error=str(e))

if __name__ == "__main__":
    main()