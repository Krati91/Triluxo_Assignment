from langchain_ollama import ChatOllama
from langchain_core.prompts import  ChatPromptTemplate, \
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


base_url = "http://localhost:11434"
model = 'llama3.2'

llm = ChatOllama(base_url=base_url, model=model)


system = SystemMessagePromptTemplate.from_template('''You are helpful AI assistant who answers user question 
                                                   based on the provided context.''')

prompt = '''Answer user question based on the provided context only! If you do not know the answer, just say 
            "I'm sorry, I didn't quite understand that. Could you rephrase?". You can accept greeting messages.
            Provide a concise answer.
        ### Context:
            {context}
            
        ### Question:
        # {question}
        
        ### Answer:'''

prompt = HumanMessagePromptTemplate.from_template(prompt)

messages = [system, prompt]
template = ChatPromptTemplate(messages)
qna_chain = template | llm | StrOutputParser()

def ask_llm(context, question):
    return qna_chain.invoke({'context':context, 
                             'question':question})