import asyncio
from pathlib import Path
from core.translator import Translator
from config import TranslationConfig
from core.api_client import ApiClient
import structlog # type: ignore

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

apiClient = ApiClient()

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
            async with Translator(config) as translator:
                if operation in ['r', 'rewrite']:
                    await apiClient.rewrite_file(input_file, user_prompt, output_pdf, output_txt)
                    print("Rewrite completed successfully!")
                else:
                    await apiClient.translate_file(input_file, output_pdf, output_txt)
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