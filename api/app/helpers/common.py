import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def length_function(text: str) -> int:
    return len(enc.encode(text))


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
