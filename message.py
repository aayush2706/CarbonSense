from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
content = input("Enter: ")
messages = [
    SystemMessage(content="You are an experienced writer, you will be given the number which is the carbon emissions predicted for the user for next month, if user keeps same lifestyle, you will write a funny but serious message for the user which will be sent to the user making him aware of the carbon emissions and how he can reduce it. The message should not exceed 150 words strictly"),
    HumanMessage(content)
]

result = model.invoke(messages)
print(result.content)   

