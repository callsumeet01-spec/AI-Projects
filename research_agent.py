import os
from ddgs import DDGS
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN was not found in the .env file.")


client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def search_web(topic, number_of_results=5):
    """Search the web and return a list of results."""
    results = []

    with DDGS() as search_engine:
        search_results = search_engine.text(
            topic,
            max_results=number_of_results
        )

        for result in search_results:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })

    return results


def create_context(results):
    """Convert search results into text for the AI model."""
    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"""
Source {index}
Title: {result['title']}
URL: {result['url']}
Description: {result['snippet']}
"""
        )

    return "\n".join(context_parts)


def summarize_research(topic, results):
    """Ask the Hugging Face model to summarize the search results."""
    context = create_context(results)

    prompt = f"""
You are a helpful personal research assistant.

Research topic:
{topic}

Search results:
{context}

Write a clear research brief using only the information in the search results.

Your response must include:
1. A short overview
2. Three to five important points
3. A limitations or caution section
4. A numbered list of source URLs

Do not invent facts. If the search results are insufficient, say so.
"""

    response = client.chat_completion(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=700,
        temperature=0.2
    )

    return response.choices[0].message.content


def run_agent():
    print("Personal Research Assistant")
    print("---------------------------")

    topic = input("What would you like to research? ")

    if not topic.strip():
        print("Please enter a research topic.")
        return

    print("\nSearching the web...")
    results = search_web(topic)

    if not results:
        print("No search results were found.")
        return

    print("Creating your research brief...\n")

    try:
        summary = summarize_research(topic, results)
        print(summary)

    except Exception as error:
        print("An error occurred:")
        print(error)


if __name__ == "__main__":
    run_agent()
