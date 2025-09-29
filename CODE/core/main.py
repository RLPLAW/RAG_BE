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