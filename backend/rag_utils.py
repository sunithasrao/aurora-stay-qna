
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant helping AuroraStay Hotels staff answer guest questions about hotel policies,
    including room types, check-in and check-out, cancellations, refunds, additional charges, parking,
    pets, amenities, guest conduct, and accessibility.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or general hotel industry practices that are not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely. If the question has multiple parts, answer every part.
    4. Do not omit important conditions, deadlines, fees, limits, exceptions, or approval requirements. Quote amounts, times, and limits exactly as written, including their units (for example, "per night", "per stay", "local property time").
    5. Focus only on information directly relevant to the user's query. If the query names a property (for example, ASH-MIA Brickell), use only that property's details.
    6. Do not include unrelated information from the retrieved policy information, such as details for other properties or rate plans that were not asked about.
    7. Answer the user's specific question directly in the first sentence, then add the supporting conditions.
    8. Keep the answer concise, clear, and easy for hotel staff to relay to a guest.
    9. Do not invent fees, refund eligibility, deadlines, room features, amenities, or exceptions.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information genuinely conflicts, clearly mention the conflict. The same table or rule appearing in more than one document is not a conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
