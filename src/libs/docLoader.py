import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from settings.settings import SOURCE_DOCUMENTS_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    # ".pdf": (PyMuPDFLoader, {}),    
    ".pdf": (PyPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext.lower() in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext.lower()]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {SOURCE_DOCUMENTS_DIRECTORY}")
    documents = load_documents(SOURCE_DOCUMENTS_DIRECTORY, ignored_files)

    if documents:
        print(f"Loaded {len(documents)} new documents from {SOURCE_DOCUMENTS_DIRECTORY}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)

        print(f"Split into {len(texts)} chunks of text (max. {CHUNK_SIZE} tokens each)")

        return texts
    else:
        print("No new documents to load")