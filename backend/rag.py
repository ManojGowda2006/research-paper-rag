from groq import Groq

from config import settings

client = Groq(api_key=settings.groq_api_key)

PROMPT_TEMPLATE = """Answer the question using only the context below. \
Cite the page(s) you used inline, like [p.3]. \
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[p.{c['page_number']}] {c['text']}" for c in retrieved_chunks
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
