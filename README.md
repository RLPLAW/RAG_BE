### NOT TESTED ON **MAC**

# MAIN REQUIREMENTS
- Install python
    - download python from the [official website](https://www.python.org/downloads/)
    - This project is currently using version **3.13.4** for **Windows**
- Install ollama
    - Download it [here](https://ollama.com/download)
    - (Optional) Installing it at a custom directory
        - Open terminal
        - Change directory to where you downloaded the ollama setup file
        - Run `OllamaSetup.exe /DIR=[CUSTOM_DIRECTORY]`
    - (Optional) Customizing where ollama save it's models when doing `ollama pull [MODEL]` from the terminal
        - Press `Win + S` and search `environment variables`and open it
        - Choose `Environment variables...` in the near bottom right corner
        - Press `New...` in the `System variables` table
        - Type **OLLAMA_MODELS** in the `Variable name:` field
        - Choose your custom directory in the `Variable value:` field
    - This project is using version **0.9.0** for **Windows**
- Chroma for storing vectors
```
docker run -d --name chroma-vectors -p 8000:8000 chromadb/chroma:latest
```

- Redis for
    - Prerequisite: install [Docker](https://www.docker.com/), port `6379` is for the **database access**, port `8001` is the **redis web UI**
```
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

<br><br>

### Prerequisite
Just paste these into the **terminal**
```
pip install sentence-transformers python-docx PyPDF2 redis langchain langchain-ollama python-dotenv langchain-community chromadb
```
```
ollama pull llama3.1
```
```
ollama pull nomic-embed-text
```

<br><br><br>

# BUGS

### Bug with `ollama serve`
`Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.`
- See which application is using port 11434
```
for /f "tokens=5" %a in ('netstat -aon ^| findstr :11434') do tasklist /FI "PID eq %a"
```
### Fix
- Temporary: Open up **terminal** and paste this in
```
set OLLAMA_HOST=127.0.0.1:11435
``` 
- Permanent: Add these to the **environment variables** like from the example above

Variable name:
```
OLLAMA_HOST
```
Variable value:
```
127.0.0.1:11435
``` 

<br><br>

# DOCUMENTS
## ChromaDB
- Check if Chroma is running, paste this URL into any **web browser**
```
localhost:8000/api/v2
```
- See what's stored in the vector database
```

```