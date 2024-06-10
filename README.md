# Ambiente Assitente CAQO
Antes de iniciar o ambiente, garanta que esteja na branch `fastapi`
```
git checkout fastapi
```

Atualize a branch com o comando `git pull`

### Subindo o ambiente
1. Subir a infra dos containers da API, Ollama e ChromaDB na pasta `api`
```
cd api/
```
2. Rode o docker compose
```
docker compose up
```
3. No navegador, acesse `localhost:5000/docs` para acessa o Swagger da API
4. Volte para a pasta `ui`
```
cd ../ui/
```
5. Rode o comando do **Streamlit** para subir a UI do chat
```
streamlit run chat.py
```
6. No navegador, acesse o chat em `localhost:8501`
