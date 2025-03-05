import os
import random
import torch
import transformers
from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from huggingface_hub import notebook_login
from datasets import load_dataset


model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

dataset = load_dataset("json", data_files="/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json")
dataset = dataset.filter(lambda x: x["dok_id"] == "H00998")

# Display options for pandas
#pd.set_option("display.max_colwidth", None)

# Login to Hugging Face if needed
notebook_login()

# Prepare documents
langchain_docs = [
    LangchainDocument(page_content=doc["anforandetext"], metadata={"source": doc["dok_id"]}) 
    for doc in tqdm(dataset["train"])
]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = []
for doc in langchain_docs:
    docs_processed += text_splitter.split_documents([doc])

QA_generation_prompt = """
                        Du måste generera frågor och svar på svenska.
                        Frågorna ska röra innehållet i debatten. 
                        Exempelvis frågor om vad olika politiker tycker om sakfrågor som diskuteras.
                        Ett exempel kan vara "Vad påstår Anders Borg om restuarangmoms?".
                        Your task is to write a factoid question and an answer given a context.
                        Your factoid question should be answerable with a specific, concise piece of factual information from the context.
                        Your factoid question should be formulated in the same style as questions users could ask in a search engine.
                        This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

                        Provide your answer as follows:

                        Output:::
                        Factoid question: (your factoid question)
                        Answer: (your answer to the factoid question)

                        Now here is the context.

                        Context: {context}\n
                        Output:::"""


N_GENERATIONS = 10  # Generate only 10 QA couples here for cost and time considerations

print(f"Generating {N_GENERATIONS} QA couples...")

outputs = []
for sampled_context in tqdm(random.sample(docs_processed, N_GENERATIONS)):
    # Generate QA couple using Hugging Face Llama model
    prompt = QA_generation_prompt.format(context=sampled_context.page_content)
    
    try:
        # Call the model with the prompt
        response = llm(prompt)[0]["generated_text"]  # Access the first generated text
        
        # Now split the response properly
        question = response.split("Faktabaserad fråga: ")[-1].split("Svar: ")[0].strip()
        answer = response.split("Svar: ")[-1].strip()

        # Ensure the answer is not too long
        assert len(answer) < 300, "Answer is too long"

        # Append the result
        outputs.append(
            {
                "context": sampled_context.page_content,
                "question": question,
                "answer": answer,
                "source_doc": sampled_context.metadata["source"],
            }
        )
    except Exception as e:
        print(f"Error generating QA pair for document {sampled_context.metadata['source']}: {e}")
        continue

print(outputs)
