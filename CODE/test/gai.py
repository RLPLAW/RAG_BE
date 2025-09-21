import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import time
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 not installed. Install with: pip install PyPDF2")
    exit(1)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
except ImportError:
    print("ReportLab not installed. Install with: pip install reportlab")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ChunkData:
    """Optimized chunk data structure"""
    content: str
    chunk_num: int
    elements: List[Dict[str, Any]]
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()[:8]

@dataclass
class TranslationConfig:
    """Configuration for translation process"""
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

class OptimizedAPITester:
    """Optimized API connectivity tester"""
    
    @staticmethod
    async def test_api_connection_async(session: aiohttp.ClientSession, api_url: str) -> Dict[str, Any]:
        """Async API connection test"""
        try:
            test_payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }
            
            async with session.post(api_url, json=test_payload, timeout=15) as response:
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
                
        except aiohttp.ClientError as e:
            return {"success": False, "error": f"Connection error: {e}"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}"}

class OptimizedDocumentStructure:
    """Optimized document structure analyzer"""
    
    def __init__(self, text: str):
        self.original_text = text
        self._lines = text.split('\n')
        self.structure = self._analyze_structure_optimized()
    
    @lru_cache(maxsize=1000)
    def _is_heading_cached(self, line: str) -> bool:
        """Cached heading detection"""
        stripped = line.strip()
        if not stripped or len(stripped) > 100:
            return False
        
        # Pre-compiled patterns for better performance
        heading_patterns = [
            re.compile(r'^[A-Z\s]{3,}$'),
            re.compile(r'^\d+\.\s+[A-Z]'),
            re.compile(r'^Chapter\s+\d+', re.IGNORECASE),
            re.compile(r'^Section\s+\d+', re.IGNORECASE),
            re.compile(r'^[IVX]+\.\s+'),
        ]
        
        for pattern in heading_patterns:
            if pattern.match(stripped):
                return True
        
        # Optimized short line detection
        if (len(stripped) < 50 and 
            not stripped.endswith(('.', '!', '?', ';', ':')) and
            len(stripped.split()) <= 8):
            return any(word[0].isupper() for word in stripped.split() if word)
        
        return False
    
    def _analyze_structure_optimized(self) -> Dict[str, Any]:
        """Optimized structure analysis with better performance"""
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
                
            elif self._is_heading_cached(line):
                # Save current paragraph before heading
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
        
        # Add final paragraph
        if current_paragraph:
            structure['paragraphs'].append({
                'content': '\n'.join(current_paragraph),
                'start_line': paragraph_start,
                'end_line': len(self._lines) - 1,
                'length': sum(len(l) for l in current_paragraph)
            })
        
        return structure
    
    def _get_heading_level(self, line: str) -> int:
        """Optimized heading level detection"""
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
        """Async context manager entry"""
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
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    async def _test_api_setup(self):
        """Test API setup during initialization"""
        test_result = await OptimizedAPITester.test_api_connection_async(
            self._session, self.config.api_url
        )
        
        if test_result["success"]:
            logger.info("API connection successful")
        else:
            logger.error(f"API connection failed: {test_result['error']}")
            raise Exception(f"API setup failed: {test_result['error']}")
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Optimized text extraction with better error handling"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        logger.info(f"Reading file: {file_path}")
        
        try:
            if path.suffix.lower() == '.pdf':
                return self._extract_from_pdf_optimized(file_path)
            elif path.suffix.lower() == '.txt':
                return self._extract_from_txt_optimized(file_path)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            raise
    
    def _extract_from_pdf_optimized(self, pdf_path: str) -> str:
        """Optimized PDF text extraction"""
        try:
            reader = PdfReader(pdf_path)
            text_parts = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Process pages in parallel
                futures = []
                for page_num, page in enumerate(reader.pages, 1):
                    future = executor.submit(self._extract_page_text, page, page_num)
                    futures.append(future)
                
                # Collect results in order
                for future in futures:
                    page_text = future.result()
                    if page_text:
                        text_parts.append(page_text)
            
            full_text = '\n\n'.join(text_parts)
            
            if not full_text.strip():
                raise ValueError("No text extracted from PDF")
            
            logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    def _extract_page_text(self, page, page_num: int) -> str:
        """Extract text from a single page"""
        try:
            extracted = page.extract_text()
            if extracted:
                return f"--- Page {page_num} ---\n\n{extracted}"
        except Exception as e:
            logger.warning(f"Error on page {page_num}: {e}")
        return ""
    
    def _extract_from_txt_optimized(self, txt_path: str) -> str:
        """Optimized text file extraction"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(txt_path, 'r', encoding=encoding, buffering=8192) as f:
                    content = f.read()
                logger.info(f"Successfully read with {encoding} encoding")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise ValueError("Could not decode text file with any supported encoding")
        
        if not content.strip():
            raise ValueError("Text file is empty")
        
        logger.info(f"Text file length: {len(content)} characters")
        return content.strip()
    
    def intelligent_chunk_optimized(self, doc_structure: OptimizedDocumentStructure) -> List[ChunkData]:
        """Optimized chunking with better memory efficiency"""
        chunks = []
        current_chunk = []
        current_length = 0
        current_elements = []
        
        # Create sorted elements list more efficiently
        all_elements = []
        
        # Add elements with line numbers for sorting
        for heading in doc_structure.structure['headings']:
            all_elements.append((heading['line'], 'heading', heading))
        
        for para in doc_structure.structure['paragraphs']:
            all_elements.append((para['start_line'], 'paragraph', para))
        
        # Sort once by line number
        all_elements.sort(key=lambda x: x[0])
        
        for line_num, element_type, element in all_elements:
            content = element['content']
            content_length = len(content)
            
            # Check if adding this element would exceed limit
            if (current_length + content_length > self.config.max_chars and 
                current_chunk):
                
                # Create chunk
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append(ChunkData(
                    content=chunk_content,
                    chunk_num=len(chunks) + 1,
                    elements=current_elements.copy()
                ))
                
                # Reset for next chunk
                current_chunk = []
                current_length = 0
                current_elements = []
            
            current_chunk.append(content)
            current_length += content_length + 2  # +2 for \n\n
            current_elements.append({
                'type': element_type,
                'content': content,
                'line': line_num
            })
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append(ChunkData(
                content=chunk_content,
                chunk_num=len(chunks) + 1,
                elements=current_elements
            ))
        
        logger.info(f"Created {len(chunks)} optimized chunks")
        return chunks
    
    async def process_chunk_async(self, chunk_data: ChunkData, operation: str = "translate", 
                                user_prompt: Optional[str] = None) -> Optional[str]:
        """Async chunk processing with semaphore for rate limiting"""
        async with self._semaphore:
            return await self._process_single_chunk(chunk_data, operation, user_prompt)
    
    async def _process_single_chunk(self, chunk_data: ChunkData, operation: str, 
                                  user_prompt: Optional[str]) -> Optional[str]:
        """Process a single chunk with retry logic"""
        logger.info(f"Processing chunk {chunk_data.chunk_num} ({len(chunk_data.content)} chars)")
        
        if operation == "translate":
            prompt = self._create_translation_prompt(chunk_data.content)
            system_msg = "You are a professional translator. Translate accurately while preserving ALL formatting, structure, headings, and paragraph breaks."
        else:  # rewrite
            prompt = self._create_rewrite_prompt(chunk_data.content, user_prompt)
            system_msg = "You are a professional rewriter. Rewrite accurately while preserving ALL formatting, structure, headings, and paragraph breaks."
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": min(len(chunk_data.content) * 4, self.config.max_tokens),
            "temperature": self.config.temperature
        }
        
        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(self.config.api_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"HTTP Error {response.status} on attempt {attempt + 1}")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                            continue
                        return None
                    
                    result = await response.json()
                    
                    if (not result.get('choices') or 
                        not result['choices'][0].get('message', {}).get('content')):
                        logger.error(f"Invalid response structure on attempt {attempt + 1}")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                            continue
                        return None
                    
                    processed_text = result['choices'][0]['message']['content']
                    logger.info(f"Chunk {chunk_data.chunk_num} processed successfully")
                    return processed_text
                    
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
        
        return None
    
    def _create_translation_prompt(self, content: str) -> str:
        """Create optimized translation prompt"""
        return f"""Translate the following text from English to Vietnamese. 
IMPORTANT: Preserve all formatting, structure, headings, and paragraph breaks exactly as they appear in the original.

Text to translate:

{content}"""
    
    def _create_rewrite_prompt(self, content: str, user_prompt: Optional[str]) -> str:
        """Create optimized rewrite prompt"""
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
        """Main async translation workflow"""
        return await self._process_file_async(input_file, "translate", None, output_pdf, output_txt)
    
    async def rewrite_file_async(self, input_file: str, user_prompt: Optional[str] = None,
                               output_pdf: Optional[str] = None, output_txt: Optional[str] = None):
        """Main async rewrite workflow"""
        return await self._process_file_async(input_file, "rewrite", user_prompt, output_pdf, output_txt)
    
    async def _process_file_async(self, input_file: str, operation: str, user_prompt: Optional[str],
                                output_pdf: Optional[str], output_txt: Optional[str]):
        """Unified async file processing workflow"""
        try:
            logger.info(f"=== Starting {operation.title()} Workflow ===")
            start_time = time.time()
            
            # Extract and analyze structure
            full_text = self.extract_text_from_file(input_file)
            logger.info(f"Extracted text length: {len(full_text)} characters")
            
            doc_structure = OptimizedDocumentStructure(full_text)
            logger.info(f"Found {len(doc_structure.structure['paragraphs'])} paragraphs, "
                       f"{len(doc_structure.structure['headings'])} headings")
            
            # Create optimized chunks
            chunks = self.intelligent_chunk_optimized(doc_structure)
            
            if len(chunks) > 10:
                logger.warning(f"Document will be split into {len(chunks)} chunks")
                proceed = input("This may take a while. Continue? (y/n): ").lower().strip()
                if proceed != 'y':
                    logger.info(f"{operation.title()} cancelled")
                    return
            
            # Process chunks concurrently
            logger.info(f"Processing {len(chunks)} chunks concurrently...")
            tasks = [
                self.process_chunk_async(chunk_data, operation, user_prompt)
                for chunk_data in chunks
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_chunks = []
            failed_chunks = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Chunk {i+1} failed with exception: {result}")
                    failed_chunks.append(i+1)
                    processed_chunks.append(f"[{operation.upper()} FAILED FOR CHUNK {i+1}]")
                elif result is None:
                    logger.error(f"Chunk {i+1} returned None")
                    failed_chunks.append(i+1)
                    processed_chunks.append(f"[{operation.upper()} FAILED FOR CHUNK {i+1}]")
                else:
                    processed_chunks.append(result)
            
            # Combine results
            full_result = "\n\n".join(processed_chunks)
            
            # Set output filenames
            input_path = Path(input_file)
            suffix = "_vietnamese" if operation == "translate" else "_rewritten"
            if output_pdf is None:
                output_pdf = f"{input_path.stem}{suffix}.pdf"
            if output_txt is None:
                output_txt = f"{input_path.stem}{suffix}.txt"
            
            # Save results
            self.save_text_backup(full_result, output_txt)
            self.create_formatted_pdf_optimized(full_result, output_pdf, doc_structure)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"=== {operation.title()} Completed ===")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"PDF output: {output_pdf}")
            logger.info(f"Text output: {output_txt}")
            
            if failed_chunks:
                logger.warning(f"Note: {len(failed_chunks)} chunks failed processing")
            
        except Exception as e:
            logger.error(f"{operation.title()} workflow failed: {e}")
            raise
    
    def save_text_backup(self, text: str, filename: str):
        """Optimized text saving with better encoding handling"""
        try:
            with open(filename, 'w', encoding='utf-8', buffering=8192) as f:
                f.write(text)
            logger.info(f"Text backup saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save text backup: {e}")
            raise
    
    def create_formatted_pdf_optimized(self, text: str, filename: str, 
                                     doc_structure: OptimizedDocumentStructure):
        """Optimized PDF creation with better performance"""
        try:
            doc = SimpleDocTemplate(
                filename, 
                pagesize=letter,
                topMargin=1*inch, 
                bottomMargin=1*inch,
                leftMargin=1*inch, 
                rightMargin=1*inch
            )
            
            # Pre-compile styles for better performance
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
            
            # Process paragraphs more efficiently
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            
            for para_text in paragraphs:
                para_text = para_text.strip()
                
                # Skip error markers
                if para_text.startswith('[') and 'FAILED' in para_text:
                    p = Paragraph(para_text, styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 12))
                    continue
                
                # Optimized heading detection
                if self._is_heading_for_pdf(para_text):
                    p = Paragraph(para_text, heading_style)
                    story.append(p)
                else:
                    # Handle paragraph with preserved line breaks
                    lines = para_text.split('\n')
                    for line in lines:
                        if line.strip():
                            p = Paragraph(line.strip(), normal_style)
                            story.append(p)
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            logger.info(f"Formatted PDF created: {filename}")
            
        except Exception as e:
            logger.error(f"PDF creation failed: {e}")
            raise
    
    @lru_cache(maxsize=500)
    def _is_heading_for_pdf(self, text: str) -> bool:
        """Cached heading detection for PDF formatting"""
        text = text.strip()
        
        if len(text) > 100 or text.endswith(('.', '!', '?')):
            return False
        
        heading_patterns = [
            re.compile(r'^[A-Z\s]{3,}$'),
            re.compile(r'^\d+\.\s+[A-Z]'),
            re.compile(r'^Chapter\s+\d+', re.IGNORECASE),
            re.compile(r'^Section\s+\d+', re.IGNORECASE),
        ]
        
        return any(pattern.match(text) for pattern in heading_patterns)

# Convenience functions for backward compatibility
async def translate_file_optimized(input_file: str, config: Optional[TranslationConfig] = None,
                                 output_pdf: Optional[str] = None, output_txt: Optional[str] = None):
    """Convenience function for file translation"""
    async with OptimizedPDFTranslator(config) as translator:
        await translator.translate_file_async(input_file, output_pdf, output_txt)

async def rewrite_file_optimized(input_file: str, user_prompt: Optional[str] = None,
                               config: Optional[TranslationConfig] = None,
                               output_pdf: Optional[str] = None, output_txt: Optional[str] = None):
    """Convenience function for file rewriting"""
    async with OptimizedPDFTranslator(config) as translator:
        await translator.rewrite_file_async(input_file, user_prompt, output_pdf, output_txt)

def main():
    """Optimized main function with async support"""
    print("=== Optimized PDF/TXT Translator ===")
    print("Enhanced version with async processing and better performance")
    
    # Get input file
    while True:
        input_file = input("\nEnter path to PDF/TXT file: ").strip()
        if not input_file:
            print("Please enter a file path.")
            continue
        
        if not Path(input_file).exists():
            print(f"File '{input_file}' not found. Please check the path.")
            continue
        
        break
    
    # Get operation type
    operation = input("Choose operation - (t)ranslate or (r)ewrite: ").lower().strip()
    if operation not in ['t', 'translate', 'r', 'rewrite']:
        operation = 't'  # Default to translate
    
    user_prompt = None
    if operation in ['r', 'rewrite']:
        user_prompt = input("Enter rewrite prompt (or press Enter for general rewrite): ").strip()
        if not user_prompt:
            user_prompt = None
    
    # Get configuration
    api_url = input("Enter API URL (press Enter for default): ").strip()
    if not api_url:
        api_url = "http://127.0.0.1:1234/v1/chat/completions"
    
    # Create configuration
    config = TranslationConfig(api_url=api_url)
    
    # Advanced settings
    advanced = input("Configure advanced settings? (y/n): ").lower().strip() == 'y'
    if advanced:
        try:
            max_concurrent = int(input(f"Max concurrent requests (default {config.max_concurrent_requests}): ").strip() or config.max_concurrent_requests)
            config.max_concurrent_requests = max_concurrent
            
            chunk_size = int(input(f"Chunk size (default {config.max_chars}): ").strip() or config.max_chars)
            config.max_chars = chunk_size
        except ValueError:
            print("Using default settings...")
    
    # Optional: custom output names
    custom_names = input("Use custom output names? (y/n): ").lower().strip() == 'y'
    output_pdf = None
    output_txt = None
    
    if custom_names:
        output_pdf = input("PDF output filename (leave empty for auto): ").strip() or None
        output_txt = input("TXT output filename (leave empty for auto): ").strip() or None
    
    async def run_translation():
        try:
            print(f"\nInitializing optimized translator with API: {config.api_url}")
            print(f"Max concurrent requests: {config.max_concurrent_requests}")
            
            if operation in ['r', 'rewrite']:
                await rewrite_file_optimized(input_file, user_prompt, config, output_pdf, output_txt)
                print("Rewrite completed successfully!")
            else:
                await translate_file_optimized(input_file, config, output_pdf, output_txt)
                print("Translation completed successfully!")
                
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            print(f"\nError: {e}")
            print("\nTroubleshooting:")
            print("1. Ensure your API server is running")
            print("2. Check the API URL is correct") 
            print("3. Verify file permissions")
            print("4. Check available disk space")
            print("5. Try reducing concurrent requests if getting rate limited")
    
    # Run the async operation
    try:
        asyncio.run(run_translation())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()