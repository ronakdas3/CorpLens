def build_rag_prompt(query, context_chunks):

    context_text = ""

    for i, chunk in enumerate(context_chunks):
        context_text += f"[{i+1}] {chunk}\n\n"

    prompt = f"""
You are a company knowledge assistant.

Answer the question ONLY using the context below.

Rules:
- Do NOT use outside knowledge
- If answer is not in context, say: "I don't know"
- Be concise and factual
- Cite sources using [1], [2], etc.

Context:
{context_text}

Question:
{query}

Answer:
"""
    return prompt