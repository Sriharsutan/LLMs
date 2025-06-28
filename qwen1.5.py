from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline

model_name = "Qwen/Qwen1.5-1.8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

hf_pipe = pipeline("text-generation", tokenizer=tokenizer, model=model, max_new_tokens=200, temperature=0.5, do_sample=True)

llm = HuggingFacePipeline(pipeline=hf_pipe)

#response = llm.invoke("explain what is natural language processing")
#print(response)

print(llm.invoke("how to solve a quadratic equation"))