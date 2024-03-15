import os
import glob
from typing import Optional, Any

from langchain.vectorstores.chroma import Chroma

# from utils.modelLoader import load_embeddings_model
from utils import modelLoader
from settings.settings import PERSIST_DIRECTORY
from libs.docLoader import process_documents


# Initialize a Chroma client &
# Load the Embeddings Model & initate the Chroma DB instance using the embeddings models
def get_chroma_db_instance(texts: Optional[Any] = None):
    embeddings = modelLoader.load_embeddings_model()

    if texts:
        db = Chroma.from_documents(
            # client=chromaClient,
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
    else:
        db = Chroma(
            # client=chromaClient,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

    return db


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    print(f"Checking if vectorstore exists at {os.path.join(persist_directory, 'index')}")
    if os.path.exists(os.path.join(persist_directory, "index")):
        if os.path.exists(
            os.path.join(persist_directory, "chroma-collections.parquet")
        ) and os.path.exists(
            os.path.join(persist_directory, "chroma-embeddings.parquet")
        ):
            list_index_files = glob.glob(os.path.join(persist_directory, "index/*.bin"))
            list_index_files += glob.glob(
                os.path.join(persist_directory, "index/*.pkl")
            )
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False



def create_document_embeddings():
    # Create embeddings

    if does_vectorstore_exist(PERSIST_DIRECTORY):
        # Update and store locally into vectorstore DB
        print(f"Appending to existing vectorstore at {PERSIST_DIRECTORY}")

        db = get_chroma_db_instance()
        collection = db.get()

        texts = process_documents([metadata["source"] for metadata in collection["metadatas"]])

        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally into vectorstore DB
        print("Creating new vectorstore")

        texts = process_documents()

        print(f"Creating embeddings. May take some minutes...")
        db = get_chroma_db_instance(texts)

    db.persist()
    db = None

    print(f"Ingestion complete! You can now query your documents")



# if __name__ == "__main__":
#     create_document_embeddings()

