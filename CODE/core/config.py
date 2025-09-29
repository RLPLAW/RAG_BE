import os
from attr import dataclass
import structlog # type: ignore

structlog.configure(
        processors = [
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True
    )

logger = structlog.get_logger()

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
        config = cls(api_url = api_url)
        
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