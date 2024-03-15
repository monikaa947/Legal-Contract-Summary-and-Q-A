from langchain.prompts import PromptTemplate
from mockData import EMPLOYEE_INFO_1, EMPLOYEE_DATA_1, EMPLOYEE_DATA_3


# ===================================================================================================================
# Define a prompt template to use with user's query, that will be used to generate answers.
# This prompt uses python f-strings as there's JSON/DICT input in the template
# ===================================================================================================================
MASTEK_RAG_PROMPT_TEMPLATE = f"""As Mastek's internal AI virtual assistant, your role is to assist company employee with various information related to company policies. Use the provided employee information, the context and not prior knowledgeto, answer the question at the end. 

Context:
{{context}}

Question: {{question}}

Base location of employee asking the question is {EMPLOYEE_DATA_1['baseLocation']}. Consider the base location and get the answer for only that location from the context.
Keep the answer breif. Don't try to make up an answer if you don't know. Don't reference any instructions or context in the answer.
Answer: 
"""
MASTEK_RAG_PROMPT = PromptTemplate(
    template=MASTEK_RAG_PROMPT_TEMPLATE, input_variables=["context", "question"]
)


# ===================================================================================================================
# Define a prompt template to use for RAG FUSION approach to generate multiple queries
# This prompt uses python f-strings as there's JSON/DICT input in the template
# ===================================================================================================================
FUSION_RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant at Mastek, a corporate company. Your task is to generate multiple search queries separated by '\n' based on a single input query. Utilize corporate terms and consider that the original query may contain words with meanings relevant to corporate jargon.

Generate multiple search queries similar to: {original_query}

OUTPUT (4 queries):
"""
FUSION_RAG_PROMPT = PromptTemplate(
    template=FUSION_RAG_PROMPT_TEMPLATE, input_variables=["original_query"]
)




# ===================================================================================================================
# A basic generalized prompt template which most probably work with any RAG based application.
# ===================================================================================================================
BASIC_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
BASIC_PROMPT = PromptTemplate(
    template=BASIC_PROMPT_TEMPLATE, input_variables=["context", "question"]
)