import numpy as np
from chunking import embeddings_model
from mmr import mmr_select


def find_similar_chunks(query: str, texts, vectors, top_k=5, lambda_param=0.5):
    query_vec = embeddings_model.embed_query(query)
    return mmr_select(
        query_vec, np.array(vectors), texts, top_k=top_k, lambda_param=lambda_param
    )
