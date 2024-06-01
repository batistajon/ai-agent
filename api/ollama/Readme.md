# Instalation

Follow the documentation the install Ollama properly: https://ollama.com/download. I this repo I'm going to install Linux version.
```
curl -fsSL https://ollama.com/install.sh | sh
```

You can check if Ollama was installed correctly
```
ollama list
```

The output should be like this
```
NAME    ID      SIZE    MODIFIED
```

Once Ollama is installed we need to pull the LLM we need - in our case LLamma3.
```
--command--
ollama pull llama3
```
```
--output--
pulling manifest
pulling 6a0746a1ec1a... 100% ▕███████████████████████████████████████████████████████████ 4.7 GB
pulling 4fa551d4f938... 100% ▕███████████████████████████████████████████████████████████ 12 KB
pulling 8ab4849b038c... 100% ▕███████████████████████████████████████████████████████████ 254 B
pulling 577073ffcc6c... 100% ▕███████████████████████████████████████████████████████████ 110 B
pulling 3f8eb4da87fa... 100% ▕███████████████████████████████████████████████████████████ 485 B
verifying sha256 digest
writing manifest
removing any unused layers
success
```
If you list it again it should show the pulled LLM
```
--command--
ollama list
```
```
--output--
NAME            ID              SIZE    MODIFIED
llama3:latest   365c0bd3c000    4.7 GB  About a minute ago
```
Now we are good to start the Ollama server, which will provide an API endpoint to access the LLM through HTTP protocol
```
ollama serve
```