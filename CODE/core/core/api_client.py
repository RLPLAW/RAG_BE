import asyncio
from pathlib import Path
import aiohttp
import structlog  # type: ignore
from typing import Dict, Any, Optional, List
from tqdm.asyncio import tqdm_asyncio
from core.core.chunking import ChunkData
from core.config import TranslationConfig
from CODE.core.utils.utils import ContentBuilder
from core.io.document_structure import DocumentStructure

contentBuilder = ContentBuilder()

# --- Logging setup ---
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class ApiClient:
    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        await self._test_api_setup()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session and not self._session.closed:
            await self._session.close()

    # API Connectivity Check
    @staticmethod
    async def test_api_connection(api_url: str, session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
        try:
            test_payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            }

            async with session.post(api_url, json=test_payload, timeout=15) as response:  # type: ignore
                if response.status != 200:
                    text = await response.text()
                    return {"success": False, "error": f"HTTP {response.status}: {text[:200]}"}

                result = await response.json()
                if "choices" not in result or not result["choices"]:
                    return {"success": False, "error": "Invalid API response structure"}

                return {
                    "success": True,
                    "response": result,
                    "test_message": result["choices"][0]["message"]["content"],
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _test_api_setup(self):
        test_result = await ApiClient.test_api_connection(self.config.api_url, self._session)
        if test_result.get("success"):
            logger.info("API connection successful")
        else:
            logger.error("API connection failed", error=test_result.get("error"))
            raise Exception(f"API setup failed: {test_result.get('error')}")

    # Chunk Processing to API
    async def process_chunk_async(
        self, chunk_data: ChunkData, operation: str = "translate", user_prompt: Optional[str] = None
    ) -> Optional[str]:
        if len(chunk_data.content) < 500:
            return (await self._process_batched_chunks([chunk_data], operation, user_prompt))[0]

        async with self._semaphore:
            return await self._process_single_chunk(chunk_data, operation, user_prompt)

    async def _process_batched_chunks(
        self, chunk_data_list: List[ChunkData], operation: str, user_prompt: Optional[str] = None
    ) -> List[Optional[str]]:
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

        # Build prompt
        if operation == "translate":
            prompt = f"""Translate the following text from English to Vietnamese. 
            Each chunk is separated by '---CHUNK_SEPARATOR---'. 
            Return the translated chunks with the same separator.
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
                {"role": "user", "content": prompt},
            ],
            "max_tokens": min(estimated_tokens * 2, self.config.max_tokens),
            "temperature": self.config.temperature,
        }

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    async with self._session.post(self.config.api_url, json=payload) as response:
                        if response.status != 200:
                            text = await response.text()
                            logger.error(
                                "Batch HTTP error", status=response.status, response_text=text[:200], attempt=attempt + 1
                            )
                            if attempt < self.config.max_retries - 1:
                                await asyncio.sleep(self.config.retry_delay)
                            continue

                        result = await response.json()
                        if not result.get("choices") or not result["choices"][0].get("message", {}).get("content"):
                            logger.error("Invalid batch response structure", response=result, attempt=attempt + 1)
                            if attempt < self.config.max_retries - 1:
                                await asyncio.sleep(self.config.retry_delay)
                            continue

                        processed_text = result["choices"][0]["message"]["content"]
                        return processed_text.split("\n\n---CHUNK_SEPARATOR---\n\n")

                except Exception as e:
                    logger.error("Batch API error", error=str(e), attempt=attempt + 1)
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    continue

            logger.error("Batch processing failed after max retries")
            return [None] * len(chunk_data_list)

    async def _process_single_chunk(
        self, chunk_data: ChunkData, operation: str, user_prompt: Optional[str]
    ) -> Optional[str]:
        if self._session is None:
            logger.error("Session not initialized")
            return None

        # Build prompt
        if operation == "translate":
            prompt = self._create_translation_prompt(chunk_data.content)
            system_msg = "You are a professional translator. Translate accurately while preserving ALL formatting."
        else:
            prompt = self._create_rewrite_prompt(chunk_data.content, user_prompt)
            system_msg = "You are a professional rewriter. Rewrite accurately while preserving ALL formatting."

        estimated_tokens = len(chunk_data.content) // 4 + 100
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": min(estimated_tokens * 2, self.config.max_tokens),
            "temperature": self.config.temperature,
        }

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(self.config.api_url, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error("HTTP error", status=response.status, response_text=text[:200], attempt=attempt + 1)
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                        continue

                    result = await response.json()
                    if not result.get("choices") or not result["choices"][0].get("message", {}).get("content"):
                        logger.error("Invalid response structure", response=result, attempt=attempt + 1)
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                        continue

                    return result["choices"][0]["message"]["content"]

            except Exception as e:
                logger.error("API error", error=str(e), attempt=attempt + 1)
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                continue

        logger.error("Processing failed after max retries")
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
    
    def intelligent_chunk(self, doc_structure: DocumentStructure, max_chunk_size: int = 2000) -> List[ChunkData]:
        """
        Create intelligent chunks based on document structure.
        Tries to keep paragraphs and sections together.
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        paragraphs = doc_structure.structure.get('paragraphs', [])
        
        for para in paragraphs:
            para_size = len(para)
            
            # If single paragraph exceeds max size, split it
            if para_size > max_chunk_size:
                if current_chunk:
                    chunks.append(ChunkData(
                        content="\n\n".join(current_chunk),
                        chunk_num=len(chunks) 
                    ))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph into sentences
                sentences = para.split('. ')
                temp_chunk = []
                temp_size = 0
                
                for sent in sentences:
                    sent_size = len(sent) + 2  # +2 for '. '
                    if temp_size + sent_size > max_chunk_size and temp_chunk:
                        chunks.append(ChunkData(
                            content='. '.join(temp_chunk) + '.',
                            chunk_num=len(chunks) 
                        ))
                        temp_chunk = []
                        temp_size = 0
                    
                    temp_chunk.append(sent)
                    temp_size += sent_size
                
                if temp_chunk:
                    chunks.append(ChunkData(
                        content='. '.join(temp_chunk),
                        chunk_num=len(chunks) 
                    ))
                continue
            
            # Normal paragraph handling
            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append(ChunkData(
                    content="\n\n".join(current_chunk),
                    chunk_num=len(chunks) 
                ))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        # Add remaining content
        if current_chunk:
            chunks.append(ChunkData(
                content="\n\n".join(current_chunk),
                chunk_num=len(chunks) 
            ))
        
        return chunks
        
    async def rewrite_file(self, input_file: str, user_prompt: Optional[str] = None,
                               output_pdf: Optional[str] = None, output_txt: Optional[str] = None):
        await self._process_file(input_file, "rewrite", user_prompt, output_pdf, output_txt)
    
    async def _process_file(self, input_file: str, operation: str, user_prompt: Optional[str],
                                output_pdf: Optional[str], output_txt: Optional[str]):
        try:
            # Determine if input is PDF or text
            input_path = Path(input_file)
            if input_path.suffix.lower() not in {'.pdf', '.txt'}:
                raise ValueError(f"Unsupported file type: {input_path.suffix}. Only .pdf and .txt are supported.")
            
            if input_path.suffix.lower() == '.pdf':
                full_text = contentBuilder.extract_from_pdf(input_file)
            else:
                full_text = contentBuilder.extract_text_from_file(input_file)
            logger.info("Text extracted", length=len(full_text))
            
            doc_structure = DocumentStructure(full_text)
            logger.info("Structure analyzed", paragraphs=len(doc_structure.structure['paragraphs']),
                       headings=len(doc_structure.structure['headings']))
            
            chunks = self.intelligent_chunk(doc_structure)
            
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
            
            # Generate output filenames
            suffix = "_vietnamese" if operation == "translate" else "_rewritten"
            if output_pdf is None:
                output_pdf = f"{input_path.stem}{suffix}.pdf"
            if output_txt is None:
                output_txt = f"{input_path.stem}{suffix}.txt"
            
            # Save outputs
            contentBuilder.create_formatted_pdf(full_result, output_pdf, doc_structure)
            
            # Optionally save as text file
            if output_txt:
                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(full_result)
                logger.info("Text file created", path=output_txt)
                            
        except Exception as e:
            logger.error("Workflow failed", error=str(e))
            raise