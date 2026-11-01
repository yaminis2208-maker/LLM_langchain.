from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.2
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer in simple English: {question}"
)

chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "question": "What is LangChain?"
})

print("LANGCHAIN OUTPUT:")
print(response)