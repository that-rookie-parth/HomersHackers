import numpy as np


def mmr_select(query_vec, doc_vecs, texts, top_k=5, lambda_param=0.5):
    selected = []
    selected_indices = []

    similarities = np.dot(doc_vecs, query_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
    )

    candidates = list(range(len(doc_vecs)))

    for _ in range(top_k):
        if not selected:
            idx = np.argmax(similarities)
            selected.append(texts[idx])
            selected_indices.append(idx)
            candidates.remove(idx)
            continue

        mmr_scores = []
        for candidate in candidates:
            sim_to_query = similarities[candidate]
            sim_to_selected = max(
                np.dot(doc_vecs[candidate], doc_vecs[i])
                / (
                    np.linalg.norm(doc_vecs[candidate]) * np.linalg.norm(doc_vecs[i])
                    + 1e-10
                )
                for i in selected_indices
            )
            mmr_score = (
                lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
            )
            mmr_scores.append((candidate, mmr_score))

        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(texts[best_idx])
        selected_indices.append(best_idx)
        candidates.remove(best_idx)

    return selected
