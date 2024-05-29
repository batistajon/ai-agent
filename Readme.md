# Assistente - IA

1. Crie um python Virtual Environment
```
python -m venv venv
```
2. Ative o virtual env
```
source bin/activate
```
3. Instale todas as dependencias
```
pip install -r requirements.txt
```
4. Crie um `.env` e cole sua chave de API da OpenAI
```
OPENAI_API_KEY=<api-key>
```
5. Rode o chainlit
```
chainlit run custom.py
```
6. Acesse `localhost:8000`