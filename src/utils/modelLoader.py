from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.llamacpp import LlamaCpp
# from langchain.llms.

from settings.settings import (
    MODEL_PATH,
    MODEL_N_CTX,
    MODEL_MAX_TKN,
    MODEL_N_BATCH,
    MODEL_SEED_VALUE,
    TEMPERATURE,
    EMBEDDINGS_MODEL_NAME,
    EMBEDDINGS_MODEL_KWARGS,
    EMBEDDINGS_ENCODE_KWARG,
)


def load_llm():
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=TEMPERATURE,
        # seed=MODEL_SEED_VALUE,
        n_ctx=MODEL_N_CTX,
        max_tokens=MODEL_MAX_TKN,
        n_batch=MODEL_N_BATCH,
        # n_gpu_layers=20,
        # model_kwargs={},
        # streaming=True,
        verbose=True,
        echo=True,
        # use_mlock=True,
        # use_mmap=True
        # repeat_penalty=1.3,
        # rope_freq_scale=1.5,
        # rope_freq_base=15000.0
    )

    return llm


def load_embeddings_model():

    hf_embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        # model_kwargs=EMBEDDINGS_MODEL_KWARGS,
        # encode_kwargs=EMBEDDINGS_ENCODE_KWARG,
    )

    return hf_embedding
