from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os

load_dotenv()
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME')


def test_ollama_model(input) -> str:
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL_NAME)
        response = llm.invoke(input)
        return response
    except Exception as e:
        print("Model failed:", e)
        return None

def test_model_response_en() -> None:
    input_text = "What is the capital of France? Answer in one short sentence."
    response = test_ollama_model(input_text)
    print("English response test:", input_text)
    

    if response:
        print("                     |_Response:", response)
    else:
        print("                     |_No Response")
        

def test_model_response_vn() -> None:
    input_text = "Thủ đô của Pháp là gì? Nói ngắn gọn trong 1 câu"
    response = test_ollama_model(input_text)
    print("Vietnamese response test:", input_text)
    
    if response:
        print("                        |_Response:", response)
    else:
        print("                        |_No Response")
    

if __name__ == "__main__":
    print("========================================================================================================================")
    print("Running basic model response test...")
    test_model_response_en()
    test_model_response_vn()
    print("========================================================================================================================")
    custom_response = test_ollama_model(input("Enter your custom question: "))
    if custom_response:
        print("Response:", custom_response)
    else:
        print("                          |_No Response")
    print("========================================================================================================================")