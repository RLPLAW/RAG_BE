import hashlib
from typing import List, Dict, Any, Self
from dataclasses import dataclass, field

from ..io.document_structure import DocumentStructure
import structlog  # type: ignore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True
)

logger = structlog.get_logger()


class ChunkConfig:
    max_chars: int = 2000


@dataclass
class ChunkData:
    content: str
    chunk_num: int
    elements: List[Dict[str, Any]] = field(default_factory=list)
    hash: str = ""
    config: ChunkConfig = field(default_factory=ChunkConfig)

    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()[:8]

    @classmethod
    def create_from_content(cls, content: str, chunk_num: int, elements: List[Dict[str, Any]] = None) -> Self:  # type: ignore
        return cls(
            content=content,
            chunk_num=chunk_num,
            elements=elements or [],
            hash=hashlib.md5(content.encode()).hexdigest()[:8]
        )
    
    def intelligent_chunk(self, doc_structure: DocumentStructure) -> List[Self]:
        """Reduce memory usage by chunking based on document structure"""
        chunks = []
        current_chunk = []
        current_length = 0
        current_elements = []
        chunk_num = 1  # Initialize chunk counter

        all_elements = sorted(
            [(h['line'], 'heading', h) for h in doc_structure.structure['headings']] 
            + [(p['start_line'], 'paragraph', p) for p in doc_structure.structure['paragraphs']],
            key=lambda x: x[0]  # Fixed: removed duplicate key declaration
        )

        for line_num, element_type, element in all_elements:
            content = element['content']
            content_length = len(content)

            if current_length + content_length > self.config.max_chars and current_chunk:
                chunks.append(self.create_from_content(
                    content='\n\n'.join(current_chunk),
                    chunk_num=chunk_num,
                    elements=current_elements
                ))
                current_chunk = []
                current_length = 0
                current_elements = []
                chunk_num += 1  # Increment chunk counter

            current_chunk.append(content)
            current_length += content_length + 2
            current_elements.append(element)

        if current_chunk:
            chunks.append(self.create_from_content(
                content='\n\n'.join(current_chunk), 
                chunk_num=chunk_num, 
                elements=current_elements
            ))

        logger.info("Generated chunks", num_chunks=len(chunks), max_chars=self.config.max_chars)
        return chunks