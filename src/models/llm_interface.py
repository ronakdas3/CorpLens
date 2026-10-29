import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer
from src.models.prompt_templates import build_rag_prompt


def load_llm():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    return tokenizer, model


def generate_simple(prompt, tokenizer, model, max_tokens=100, temperature=0.1):
    """
    Fast direct text generation for agent decisions (routing, evaluation, rewriting).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.1),
        top_p=0.9
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return result


def generate_answer(query, context_chunks, tokenizer, model, max_tokens=300):
    """
    Synchronous answer generation using retrieved context.
    """
    prompt = build_rag_prompt(query, context_chunks)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.2,
        do_sample=True,
        top_p=0.9
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer


def generate_answer_stream(query, context_chunks, tokenizer, model, max_tokens=300):
    """
    Streaming answer generation yielding tokens in real time.
    """
    prompt = build_rag_prompt(query, context_chunks)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=0.2,
        do_sample=True,
        top_p=0.9
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    clean_answer = []
    found_answer_prefix = False

    for new_text in streamer:
        # Filter out repetitive 'Answer:' prefix if model produces it
        if not found_answer_prefix and "Answer:" in "".join(clean_answer) + new_text:
            found_answer_prefix = True
            parts = ("".join(clean_answer) + new_text).split("Answer:")
            clean_answer = []
            yield parts[-1].lstrip()
            continue
        clean_answer.append(new_text)
        yield new_text

    thread.join()
