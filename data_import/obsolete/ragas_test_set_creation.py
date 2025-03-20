import json
import os
from dotenv import load_dotenv
from datasets  import load_dataset
from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
N_GENERATIONS = 5  # Generate only 5 QA pairs for cost and time considerations
#dataset = load_dataset("json", data_files="/mnt/c/Users/User/thesis/data_import/exp1/even_more_filtered_riksdag.json", split="train")
#dataset = dataset.filter(lambda x: x["dok_id"] in ["H90968", "H90982"])
url = "https://api.openai.com/v1/chat/completions"
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

path = "/mnt/c/Users/User/thesis/data_import/exp1/example_data/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)


output_path = "/mnt/c/Users/User/thesis/data_import/exp1/generated_testset.json"
dataset = dataset.to_list()  # Ensure Testset has this method

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"Dataset saved to {output_path}")