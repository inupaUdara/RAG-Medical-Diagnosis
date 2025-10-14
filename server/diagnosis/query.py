import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rbac-diagnosis-index-openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud=os.getenv("PINECONE_ENV", "us-east-1"), region=os.getenv("PINECONE_ENV", "us-east-1"))

# Models via env with sensible defaults
EMBEDDING_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
EMBEDDING_DIM = int(os.getenv("OPENAI_EMBED_DIM", _MODEL_DIMS.get(EMBEDDING_MODEL, 1536)))

def _get_or_create_index(pc, base_name: str, dim: int, metric: str = "dotproduct"):
    existing = [i.name for i in pc.list_indexes()]
    target = base_name
    existing_dim = None
    if base_name in existing:
        try:
            desc = pc.describe_index(base_name)
            existing_dim = getattr(desc, "dimension", None)
            if not existing_dim:
                spec_obj = getattr(desc, "spec", None)
                if isinstance(spec_obj, dict):
                    existing_dim = spec_obj.get("dimension")
        except Exception:
            existing_dim = None
        if existing_dim and existing_dim != dim:
            target = f"{base_name}-{dim}d"
    if target not in existing:
        pc.create_index(name=target, dimension=dim, metric=metric, spec=spec)
        while not pc.describe_index(target).status["ready"]:
            time.sleep(1)
    return pc.Index(target), target

index, PINECONE_INDEX_NAME = _get_or_create_index(pc, PINECONE_INDEX_NAME, EMBEDDING_DIM)

embed_model=OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm=ChatOpenAI(model=CHAT_MODEL, temperature=0)

prompt=PromptTemplate.from_template(
    """
You are a medical assistant. Using only the provided context (portions of the user's report), produce:
1) A concise probable diagnosis (1-2 lines)
2) Key findings from the report (bullet points)
3) Recommended next steps (tests/treatments) — label clearly as suggestions, not medical advice.

Context:
{context}

User question:
{question}
""")

rag_chain=prompt | llm

async def diagnosis_report(user:str,doc_id:str,question:str):
    try:
        # embed question
        embedding=await asyncio.to_thread(embed_model.embed_query,question)
        # query pinecone with metadata filter (v3-friendly)
        results=await asyncio.to_thread(
            index.query,
            vector=embedding,
            top_k=5,
            include_metadata=True,
            filter={"doc_id": doc_id}
        )

        # robustly access matches from response
        matches = []
        try:
            matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", []) or []
        except Exception:
            matches = []

        contexts=[]
        sources_set=set()
        for match in matches:
            md = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {}) or {}
            text_snippet = md.get("text") or ""
            if text_snippet:
                contexts.append(text_snippet)
            src = md.get("source")
            if src:
                sources_set.add(src)

        if not contexts:
            return {"diagnosis": None, "explanation": "No report content indexed for this doc_id yet. Please retry after a moment."}

        # limit context length
        context_text = "\n\n".join(contexts[:5])

        # final call the rag chain
        final=await asyncio.to_thread(rag_chain.invoke,{"context":context_text,"question":question})
        content = getattr(final, "content", str(final))
        return {"diagnosis": content, "sources": list(sources_set)}
    except Exception as e:
        return {"diagnosis": None, "explanation": f"Error generating diagnosis: {str(e)}"}