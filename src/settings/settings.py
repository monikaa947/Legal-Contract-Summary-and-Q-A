from dotenv import load_dotenv
load_dotenv()


# Path to LLM Model
MODEL_PATH="D:\models\mistral-7b-instruct-v0.2.Q5_K_M.gguf"

# Model's context window size - in number of tokens
MODEL_N_CTX=4096

# Max no. of tokens to generate in the output
MODEL_MAX_TKN=1024

# Token to process in parallel
MODEL_N_BATCH=512

# Temperature setting for the LLM model - Higher number gives more creative content, lower number gives more factual content
TEMPERATURE=0.7

# Random seed value to be used during output generation
MODEL_SEED_VALUE=12533

# Size of a text chunk to be created during text splitting & chunk overlap parameter
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

# Define the directory for storing vectore database
PERSIST_DIRECTORY = "db"

# Directory where document which will act as a datasource will be stored
SOURCE_DOCUMENTS_DIRECTORY = "./source_documents"

# HuggingFace Embeddings model name which will be used to embed the text into vectors
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBEDDINGS_MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"
EMBEDDINGS_MODEL_KWARGS = {'device': 'cpu'}
EMBEDDINGS_ENCODE_KWARG = {'normalize_embeddings': False}