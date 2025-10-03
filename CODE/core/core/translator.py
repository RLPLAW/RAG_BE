from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
from typing import Optional, Dict, Any, Tuple
import aiohttp
from charset_normalizer import detect
from pypdf import PdfReader
import structlog # type: ignore
from core.api_client import ApiClient
from config import TranslationConfig

apiClient = ApiClient()

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

class Translator:

    def __init__(self, config: TranslationConfig):
        self.config = apiClient.config
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=apiClient.config.request_timeout)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        await apiClient._test_api_setup()
        return self
    
    @lru_cache(maxsize=100)
    def _detect_encoding(self, file_path: str) -> Optional[str]:
        """Detect file encoding using charset-normalizer."""
        try:
            with open(file_path, "rb") as f:
                result: Dict[str, Any] = detect(f.read(4096))  # pyright: ignore[reportAssignmentType] # Read 4KB for detection

                if result.get("encoding") and result.get("confidence", 0.0) > 0.8:
                    logger.debug(
                        "Encoding detected for %s: %s (confidence=%.2f)",
                        file_path,
                        result["encoding"],
                        result["confidence"],
                    )
                    return result["encoding"]

                logger.debug(
                    "Encoding detection not reliable for %s: %s (confidence=%.2f)",
                    file_path,
                    result.get("encoding"),
                    result.get("confidence", 0.0),
                )
                return None

        except Exception as e:
            logger.warning("Encoding detection failed for %s: %s", file_path, str(e))
            return None

    def extract_text_from_file(self, file_path: str) -> str:
        encodings = [
            self._detect_encoding(file_path),
            "utf-8",
            "windows-1258",
            "latin-1",
            "iso-8859-1",
        ]
        encodings = [e for e in encodings if e]  # Remove None

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding, buffering=8192) as f:
                    content = f.read()
                logger.info("Text extracted from %s", file_path)
                if not content.strip():
                    raise ValueError("Text file is empty")
                return content.strip()
            except Exception as e:
                logger.debug("Encoding failed for %s: %s", encoding, str(e))
                continue

        raise ValueError(f"Failed to extract text from {file_path}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
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
                    logger.warning("Page extraction error", 
                                   page_num+1, 
                                   str(e))
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

            logger.info("PDF extracted", len(full_text), num_pages)
            return full_text.strip()
        except Exception as e:
            logger.error("Failed to extract text from PDF", str(e))
            raise

    