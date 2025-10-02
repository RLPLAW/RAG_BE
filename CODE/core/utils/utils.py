import os
import structlog  # type: ignore
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pypdf import PdfReader
from chardet import detect # type: ignore
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from io_utils.document_structure import DocumentStructure

# Proper structlog configuration
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


class ContentBuilder:
    
    @lru_cache(maxsize=100)
    def _detect_encoding(self, file_path: str) -> Optional[str]:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                result: Dict[str, Any] = detect(f.read(4096))  # Read 4KB for detection
                if result['encoding'] and result.get('confidence', 0.0) > 0.8:
                    logger.debug("Encoding detected", file=file_path, encoding=result['encoding'], confidence=result['confidence'])
                    return result['encoding']
                logger.debug("Encoding not reliable", file=file_path, encoding=result.get('encoding'), confidence=result.get('confidence', 0.0))
                return None
        except Exception as e:
            logger.warning("Encoding detection failed", file=file_path, error=str(e))
            return None

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from a text file with intelligent encoding detection."""
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
    
    def extract_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using parallel processing for better performance.
        Note: Method renamed from _extract_from_pdf to extract_from_pdf (public).
        """
        try:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            max_workers = min(os.cpu_count() * 2, num_pages) if num_pages > 10 else 1  # type: ignore
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

    def create_formatted_pdf(self, text: str, filename: str,
                             doc_structure: DocumentStructure):
        """
        Create a formatted PDF from text with proper styling.
        Fixed: Removed incorrect 'Translator' parameter.
        """
        try:
            # Register Unicode font for Vietnamese support
            pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))

            # Create document
            doc = SimpleDocTemplate(
                filename, 
                pagesize=letter,
                topMargin=1*inch, 
                bottomMargin=1*inch,
                leftMargin=1*inch, 
                rightMargin=1*inch
            )

            styles = getSampleStyleSheet()
            
            # Define custom styles
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
            
            # Split text into paragraphs
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            
            for para_text in paragraphs:
                para_text = para_text.strip()
                
                # Handle error messages
                if para_text.startswith('[') and 'FAILED' in para_text:
                    p = Paragraph(para_text, styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 12))
                    continue
                
                # Check if it's a heading
                if DocumentStructure._is_heading(para_text):
                    p = Paragraph(para_text, heading_style)
                    story.append(p)
                else:
                    # Handle regular paragraphs
                    lines = para_text.split('\n')
                    for line in lines:
                        if line.strip():
                            p = Paragraph(line.strip(), normal_style)
                            story.append(p)
                    story.append(Spacer(1, 6))
            
            # Build the PDF
            doc.build(story)
            logger.info("PDF created successfully", filename=filename)
            
        except Exception as e:
            logger.error("PDF creation failed", error=str(e))
            raise