import numpy as np
import pandas as pd
import google.generativeai as genai
from decouple import config

from documents.documents_list import documents


genai.configure(api_key=config("GOOGLE_API_KEY"))

for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)

# Exemplo de embedding
title = "The quick brown fox jumps over the lazy dog."
sample_text = "Título: The quick brown fox jumps over the lazy dog."
"\n"
"Artigo completo:\n"
"\n"
"Gemini API Studio: Maneira acessível de criar e treinar modelos de IA"

embedding = genai.embed_content(
    model="models/embedding-001",
    content=sample_text,
    title=title,
    task_type="RETRIEVAL_DOCUMENT",
)


df = pd.DataFrame(documents)
df.columns = ["Titulo", "Conteudo"]

# Embedding dos documentos
model = "models/embedding-001"


def get_embedding(title, text):
    return genai.embed_content(
        model=model, content=text, title=title, task_type="RETRIEVAL_DOCUMENT"
    )["embedding"]


df["Embeddings"] = df.apply(
    lambda row: get_embedding(row["Titulo"], row["Conteudo"]), axis=1
)


def generate_search_querys(consulta, base, model):
    embedding_da_consulta = genai.embed_content(
        model=model, content=consulta, task_type="RETRIEVAL_QUERY"
    )["embedding"]

    produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta)

    indice = np.argmax(produtos_escalares)
    return df.iloc[indice]["Conteudo"]


consulta = "Me mostre como passa marcha "

trecho = generate_search_querys(consulta, df, model)


generation_config = {"temperature": 0.5, "candidate_count": 1}

prompt = f"Reescreva esse texto de uma forma mais descontraída, sem adicionar informações que não façam parte do texto: {trecho}"

model_2 = genai.GenerativeModel("gemini-1.0-pro", generation_config=generation_config)
response = model_2.generate_content(prompt)
print(response.text)
