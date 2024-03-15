from libs.docEmbeddings import create_document_embeddings
from libs.vectorSearch import vector_search

if __name__ == '__main__':    
    # Creates document embeddings using HuggingFace Embeddings models and stores them in ChromaDB
    create_document_embeddings()

    query = "Give me a quick summary of this consulting agreement contract between Kiromic, Inc & Gianluca Rotino"
    search_results = vector_search(query)
    print('='*100)
    # print(type(search_results))
    # print(search_results.keys())
    # print(search_results.values())
    for key in search_results:
        print(f"**{search_results[key]}:\n==>{key}\n\n")
        print('-'*100)

