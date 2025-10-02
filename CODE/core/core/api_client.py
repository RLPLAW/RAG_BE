import asyncio
from pathlib import Path
import aiohttp
import structlog
from typing import Dict, Any, Optional, List
from tqdm.asyncio import tqdm_asyncio
from core.chunking import ChunkData
from config import TranslationConfig
from utils.utils import ContentBuilder
from io_utils.document_structure import DocumentStructure

content_builder = ContentBuilder()
logger = structlog.get_logger()

class ApiClient:
    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    async def __aenter__(self):
        await self._initialize_session()
        await self._test_api_setup()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _initialize_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5),
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                headers={"Content-Type": "application/json"}
            )

    @staticmethod
    async def test_api_connection(api_url: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            async with session.post(
                api_url,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10
                },
                timeout=15
            ) as response:
                if response.status != 200:
                    return {"success": False, "error": f"HTTP {response.status}: {(await response.text())[:200]}"}
                result = await response.json()
                if "choices" not in result or not result["choices"]:
                    return {"success": False, "error": "Invalid API response structure"}
                return {
                    "success": True,
                    "response": result,
                    "test_message": result["choices"][0]["message"]["content"]
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _test_api_setup(self):
        await self._initialize_session()
        if not self._session:
            logger.error("Session initialization failed")
            raise Exception("API session initialization failed")
        test_result = await self.test_api_connection(self.config.api_url, self._session)
        if test_result["success"]:
            logger.info("API connection successful")
        else:
            logger.error("API connection failed", error=test_result["error"])
            raise Exception(f"API setup failed: {test_result['error']}")

    async def process_chunk_async(
        self, chunk_data: ChunkData, operation: str = "translate", user_prompt: Optional[str] = None
    ) -> Optional[str]:
        await self._initialize_session()
        if len(chunk_data.content) < 500:
            return (await self._process_batched_chunks([chunk_data], operation, user_prompt))[0]
        async with self._semaphore:
            return await self._process_single_chunk(chunk_data, operation, user_prompt)

    async def _process_batched_chunks(
        self, chunk_data_list: List[ChunkData], operation: str, user_prompt: Optional[str] = None
    ) -> List[Optional[str]]:
        await self._initialize_session()
        if not self._session or not chunk_data_list:
            logger.error("Invalid batch processing input", session_initialized=bool(self._session), chunks=len(chunk_data_list))
            return [None] * len(chunk_data_list)

        # Fix: Extract content strings from ChunkData objects
        content_parts = []
        for c in chunk_data_list:
            content = c.content
            # Handle case where content might be a dict or other non-string type
            if isinstance(content, dict):
                logger.warning("Content is dict, converting to string", chunk_data=content)
                content = str(content)
            elif not isinstance(content, str):
                logger.warning("Content is not string, converting", content_type=type(content).__name__)
                content = str(content)
            content_parts.append(content)
        
        combined_content = "\n\n---CHUNK_SEPARATOR---\n\n".join(content_parts)
        
        if not combined_content.strip():
            logger.error("Empty combined content")
            return [None] * len(chunk_data_list)

        prompt, system_msg = (
            (
                f"Translate the following text from English to Vietnamese. Each chunk is separated by '---CHUNK_SEPARATOR---'. "
                f"Preserve ALL formatting, structure, headings, and paragraph breaks.\n\nText to translate:\n\n{combined_content}",
                "You are a professional translator. Translate accurately while preserving ALL formatting."
            ) if operation == "translate" else (
                f"Rewrite the following text. Each chunk is separated by '---CHUNK_SEPARATOR---'. "
                f"Return the rewritten chunks with the same separator. Preserve ALL formatting.\n\nText to rewrite: {combined_content}",
                "You are a professional rewriter. Rewrite accurately while preserving ALL formatting."
            )
        )

        return await self._make_api_request(combined_content, prompt, system_msg, len(chunk_data_list))

    async def _process_single_chunk(
        self, chunk_data: ChunkData, operation: str, user_prompt: Optional[str]
    ) -> Optional[str]:
        await self._initialize_session()
        if not self._session or self._session.closed:
            logger.error("Session not initialized or closed")
            return None

        prompt = (
            self._create_translation_prompt(chunk_data.content) if operation == "translate"
            else self._create_rewrite_prompt(chunk_data.content, user_prompt)
        )
        system_msg = (
            "You are a professional translator. Translate accurately while preserving ALL formatting."
            if operation == "translate"
            else "You are a professional rewriter. Rewrite accurately while preserving ALL formatting."
        )

        return (await self._make_api_request(chunk_data.content, prompt, system_msg, 1))[0]

    async def _make_api_request(
        self, content: str, prompt: str, system_msg: str, chunk_count: int
    ) -> List[Optional[str]]:
        await self._initialize_session()
        if not self._session or self._session.closed:
            logger.error("Session not initialized or closed")
            return [None] * chunk_count

        estimated_tokens = len(content) // 4 + 100
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": min(estimated_tokens * 2, self.config.max_tokens),
            "temperature": self.config.temperature
        }

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(self.config.api_url, json=payload) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        logger.error("HTTP error", 
                                    status=response.status, 
                                    response_text=response_text[:500], 
                                    attempt=attempt + 1)
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                        continue
                    
                    try:
                        result = await response.json()
                    except Exception as json_error:
                        logger.error("JSON parse error", 
                                    error=str(json_error), 
                                    response_text=response_text[:500],
                                    attempt=attempt + 1)
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                        continue
                    
                    if not result.get("choices") or not result["choices"][0].get("message", {}).get("content"):
                        logger.error("Invalid response structure", 
                                    response=result, 
                                    attempt=attempt + 1)
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                        continue
                    
                    processed_text = result["choices"][0]["message"]["content"]
                    logger.info("API request successful", 
                               content_length=len(processed_text),
                               chunk_count=chunk_count)
                    
                    if chunk_count > 1:
                        split_results = processed_text.split("\n\n---CHUNK_SEPARATOR---\n\n")
                        if len(split_results) != chunk_count:
                            logger.warning("Chunk count mismatch",
                                          expected=chunk_count,
                                          received=len(split_results))
                        return split_results
                    else:
                        return [processed_text]
                        
            except asyncio.TimeoutError:
                logger.error("Request timeout", attempt=attempt + 1)
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                continue
            except Exception as e:
                logger.error("API error", 
                            error=str(e), 
                            error_type=type(e).__name__,
                            attempt=attempt + 1)
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                continue

        logger.error("Processing failed after max retries", 
                    max_retries=self.config.max_retries)
        return [None] * chunk_count

    @staticmethod
    def _create_translation_prompt(content: str) -> str:
        return (
            f"Translate the following text from English to Vietnamese. "
            f"Preserve all formatting, structure, headings, and paragraph breaks exactly.\n\n"
            f"Text to translate:\n\n{content}"
        )

    @staticmethod
    def _create_rewrite_prompt(content: str, user_prompt: Optional[str]) -> str:
        return (
            f"Find and rewrite content similar to '{user_prompt}'. If not found, keep unchanged. "
            f"Preserve all formatting, structure, headings, and paragraph breaks exactly.\n\n"
            f"Text to rewrite: {content}" if user_prompt else
            f"Rewrite and improve the following text while maintaining its original meaning. "
            f"Preserve all formatting, structure, headings, and paragraph breaks exactly.\n\n"
            f"Text to rewrite: {content}"
        )

    def intelligent_chunk(self, doc_structure: DocumentStructure, max_chunk_size: int = 2000) -> List[ChunkData]:
        chunks = []
        current_chunk = []
        current_size = 0
        paragraphs = doc_structure.structure.get('paragraphs', [])

        for para_dict in paragraphs:
            # Fix: Extract the 'content' string from the paragraph dictionary
            para = para_dict['content'] if isinstance(para_dict, dict) else para_dict
            
            para_size = len(para)
            if para_size > max_chunk_size:
                if current_chunk:
                    chunks.append(ChunkData(content="\n\n".join(current_chunk), chunk_num=len(chunks)))
                    current_chunk = []
                    current_size = 0

                sentences = para.split('. ')
                temp_chunk = []
                temp_size = 0
                for sent in sentences:
                    sent_size = len(sent) + 2
                    if temp_size + sent_size > max_chunk_size and temp_chunk:
                        chunks.append(ChunkData(content='. '.join(temp_chunk) + '.', chunk_num=len(chunks)))
                        temp_chunk = []
                        temp_size = 0
                    temp_chunk.append(sent)
                    temp_size += sent_size
                if temp_chunk:
                    chunks.append(ChunkData(content='. '.join(temp_chunk), chunk_num=len(chunks)))
                continue

            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append(ChunkData(content="\n\n".join(current_chunk), chunk_num=len(chunks)))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append(ChunkData(content="\n\n".join(current_chunk), chunk_num=len(chunks)))

        return chunks

    async def rewrite_file(self, input_file: str, user_prompt: Optional[str] = None,
                          output_pdf: Optional[str] = None, output_txt: Optional[str] = None):
        await self._process_file(input_file, "rewrite", user_prompt, output_pdf, output_txt)

    async def translate_file(self, input_file: str, output_pdf: Optional[str] = None,
                            output_txt: Optional[str] = None):
        await self._process_file(input_file, "translate", None, output_pdf, output_txt)

    async def _process_file(self, input_file: str, operation: str, user_prompt: Optional[str],
                           output_pdf: Optional[str], output_txt: Optional[str]):
        await self._initialize_session()
        try:
            input_path = Path(input_file)
            if input_path.suffix.lower() not in {'.pdf', '.txt'}:
                raise ValueError(f"Unsupported file type: {input_path.suffix}. Only .pdf and .txt supported.")

            full_text = (
                content_builder.extract_from_pdf(input_file)
                if input_path.suffix.lower() == '.pdf'
                else content_builder.extract_text_from_file(input_file)
            )
            logger.info("Text extracted", length=len(full_text))

            doc_structure = DocumentStructure(full_text)
            logger.info("Structure analyzed", paragraphs=len(doc_structure.structure['paragraphs']),
                        headings=len(doc_structure.structure['headings']))

            chunks = self.intelligent_chunk(doc_structure)
            if len(chunks) > 10:
                logger.warning("Large document", chunk_count=len(chunks))
                if input("This may take a while. Continue? (y/n): ").lower().strip() != 'y':
                    logger.info("Operation cancelled")
                    return

            logger.info("Processing chunks", total=len(chunks))
            results = await tqdm_asyncio.gather(
                *[self.process_chunk_async(chunk, operation, user_prompt) for chunk in chunks],
                desc="Processing chunks"
            )

            processed_chunks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception) or result is None:
                    logger.error("Chunk failed", chunk_num=i+1, error=str(result) if isinstance(result, Exception) else "None")
                    processed_chunks.append(f"[{operation.upper()} FAILED FOR CHUNK {i+1}]")
                else:
                    processed_chunks.append(result)

            full_result = "\n\n".join(processed_chunks)
            suffix = "_vietnamese" if operation == "translate" else "_rewritten"
            output_pdf = output_pdf or f"{input_path.stem}{suffix}.pdf"
            output_txt = output_txt or f"{input_path.stem}{suffix}.txt"

            content_builder.create_formatted_pdf(full_result, output_pdf, doc_structure)
            if output_txt:
                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(full_result)
                logger.info("Text file created", path=output_txt)

        except Exception as e:
            logger.error("Workflow failed", error=str(e))
            raise