import os
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from ..config.db import reports_collection
from typing import List
from fastapi import UploadFile

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rbac-diagnosis-index-openai")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_reports")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.makedirs(UPLOAD_DIR, exist_ok=True)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
EMBEDDING_DIM = int(os.getenv("OPENAI_EMBED_DIM", _MODEL_DIMS.get(EMBEDDING_MODEL, 1536)))

# initialize pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

# Resolve index ensuring dimension compatibility; if the base exists with a different dim,
# use a suffixed name like "<base>-1536d" and create it if necessary.
def _get_or_create_index(pc, base_name: str, dim: int, metric: str = "dotproduct"):
    existing = [i.name for i in pc.list_indexes()]
    target = base_name
    existing_dim = None
    if base_name in existing:
        try:
            desc = pc.describe_index(base_name)
            # Try multiple shapes to read dimension safely
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


async def load_vectorstore(uploaded_files:List[UploadFile],uploaded:str,doc_id:str):
    """
        Save files, chunk texts, embed texts, upsert in Pinecone and write metadata to Mongo
    """
    embed_model=OpenAIEmbeddings(model=EMBEDDING_MODEL)

    saved_files=[]
    for file in uploaded_files:
        filename=Path(file.filename).name
        save_path=Path(UPLOAD_DIR)/ f"{doc_id}_{filename}"
        content=await file.read()
        with open(save_path,"wb") as f:
            f.write(content)
        saved_files.append((filename, save_path))

    # process all saved PDFs
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    all_texts=[]
    all_ids=[]
    all_metadatas=[]
    idx=0

    for filename, save_path in saved_files:
        try:
            loader=PyPDFLoader(str(save_path))
            documents=loader.load()
        except Exception:
            # skip non-PDF or unreadable files quietly
            continue

        chunks= splitter.split_documents(documents)
        texts=[chunk.page_content for chunk in chunks]
        ids=[f"{doc_id}-{i}" for i in range(idx, idx+len(chunks))]
        metadatas=[
            {
                "source": filename,
                "doc_id": doc_id,
                "uploader": uploaded,
                "page": chunk.metadata.get("page", None),
                "text": chunk.page_content[:2000]
            }
            for chunk in chunks
        ]

        all_texts.extend(texts)
        all_ids.extend(ids)
        all_metadatas.extend(metadatas)
        idx += len(chunks)

    # persist report metadata even if zero chunks (helps debugging)
    reports_collection.insert_one({
        "doc_id": doc_id,
        "filename": ", ".join([name for name, _ in saved_files]) if saved_files else None,
        "uploader": uploaded,
        "num_chunks": len(all_texts),
        "uploaded_at": time.time()
    })

    if not all_texts:
        return  # nothing to index

    # get embeddings in a thread
    embeddings=await asyncio.to_thread(embed_model.embed_documents, all_texts)

    # upsert using Pinecone v3 format
    vectors=[{"id": v_id, "values": vec, "metadata": md}
             for v_id, vec, md in zip(all_ids, embeddings, all_metadatas)]

    def upsert():
        index.upsert(vectors=vectors)

    await asyncio.to_thread(upsert)