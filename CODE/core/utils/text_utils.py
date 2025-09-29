from sympy import Si
from core.io.document_structure import DocumentStructure
from core.core.translator import Translator
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
import structlog # type: ignore

structlog = (
    structlog.configure(
        processors=[
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True
    )
)

logger = structlog.get_logger() 

class ContentBuilder:

    def create_formatted_pdf(self, Translator, text: str, filename: str,
                        doc_structure: DocumentStructure):
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
                
                if DocumentStructure._is_heading(para_text):
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
        except Exception as e:
            logger.error("PDF creation failed", error=str(e))
            raise