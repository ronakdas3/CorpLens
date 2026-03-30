def build_rag_prompt(query, context_chunks):

    context_text = ""

    for i, chunk in enumerate(context_chunks):
        context_text += f"[{i+1}] {chunk}\n\n"

    prompt = f"""
You are a research assistant.

Answer the question using the context.

- Give a clear and complete answer.
- Use citations like [1], [2] if helpful.
- Do NOT say "answer not found" unless absolutely no information is available.
- Try your best using the context.

Context:
{context_text}

Question:
{query}

Answer:
"""

    return prompt