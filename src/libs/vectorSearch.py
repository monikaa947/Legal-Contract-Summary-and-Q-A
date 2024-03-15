from libs.docEmbeddings import get_chroma_db_instance


vectorDB = get_chroma_db_instance()
retriever = vectorDB.as_retriever()

# Set a retrieval method that sets a similarity score threshold and only returns documents with a score above that threshold.
# retriever = vectorDB.as_retriever(
#     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
# )


# If the underlying vector store supports maximum marginal relevance search, you can specify that as the search type.
# TODO: Lookup mmr
# retriever = vectorDB.as_retriever(search_type="mmr")


# Fetch documents using Similarity Search
def vector_search(query):
    search_results = {}
    # similary search
    retrieved_docs = retriever.get_relevant_documents(query)
    print('-'*100)
    print("retrieved_docs: ", retrieved_docs)
    print('-'*100)
    for i in retrieved_docs:
        search_results[i.page_content] = f"From page {i.metadata['page']} of {i.metadata['source']}"

    # print(f"\nRelavant document founds for {query}:\n{search_results}")
    return search_results


# if __name__ == "__main__":

#     query = "Give me a quick summary of this consulting agreement contract between Kiromic, Inc & Gianluca Rotino"
#     search_results = vector_search(query)
#     print('='*100)
#     for key, value in search_results:
#         print(f"**{value}:\n==>{key}\n\n")
