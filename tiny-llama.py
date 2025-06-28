from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = PromptTemplate.from_template("Q: {question}\nA:")

chain = prompt | llm

response = chain.invoke({"question": "Suggest me a name of mexican restaruant"})
print(response)
