import os
import random
import re
import openai
import json
import time
import requests


class QueryGenerator:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.OPENAI_API_KEY}"
        }
        self.model = "gpt-4o"
        self.max_tokens = 500
        self.temperature = 0
            
    def generate_qa_inference_person(self, entry):
        qa_type = "generate_qa_inference_person"
        prompt = f"""
                  You are an expert in analyzing Swedish parliamentary debates and generating relevant questions and answers. 
                    
                    Based on the speech below, create a **question in Swedish** that asks about the position of the speaker ({entry['talare']}) as expressed in the speech.
                    Create an **answer in Swedish** based on the question you just created                    
                    Guidelines:
                    - Do **not** copy the exact wording from the speech — rephrase using your own words while preserving the intended meaning.
                    - Focus on **broader political opinions or positions**, not on minor or irrelevant details (e.g., do not ask questions like *“Why doesn’t X read the news about herself?”*).
                    - Assume that these are **statements by politicians**, representing **opinions or positions**, not necessarily absolute facts.
                    - Both the question and the answer must be written in **Swedish**, and must be directly relevant to the content of the speech.
                    - The question should start with something like: *"Hur argumenterar {entry['talare']} för..." or "Vad tycker {entry['talare']} om..."*

                    ### Speech ###
                    {entry['anforandetext']}
                    ##################

                    **Output format:**  
                    {{
                        "question": "<Generated question in Swedish>",
                        "answer": "<Generated answer based on the speech in Swedish>"
                    }}
                    """
        return self._query_llm(entry, prompt, qa_type)


    def generate_qa_inference_party(self, entry):
        qa_type = "generate_qa_inference_party"
        prompt = f"""                    
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant questions and answers. 
                    
                    Based on the speech below, create a **question in Swedish** that asks about the position of the party ({entry['parti']}) as expressed in the speech.
                    Create an **answer in Swedish** based on the question you just created                    
                    Guidelines:
                    - Use the **full name of the party**, not the abbreviation (see list below).
                    - Do **not** copy the exact wording from the speech — rephrase using your own words while preserving the intended meaning.
                    - Focus on **broader political opinions or positions**, not on minor or irrelevant details (e.g., do not ask questions like *“Why doesn’t X read the news about herself?”*).
                    - Assume that these are **statements by politicians**, representing **opinions or positions**, not necessarily absolute facts.
                    - Both the question and the answer must be written in **Swedish**, and must be directly relevant to the content of the speech.
                    - The question should start with something like: *"Hur argumenterar {entry['parti']} för..." or "Vad tycker {entry['parti']} om..."*

                    **Party abbreviations and their full names:**
                    M: Moderaterna  
                    S: Socialdemokraterna  
                    SD: Sverigedemokraterna  
                    C: Centerpartiet  
                    V: Vänsterpartiet  
                    L: Liberalerna  
                    KD: Kristdemokraterna  
                    MP: Miljöpartiet


                    ### Speech ###
                    {entry['anforandetext']}
                    ##################

                    **Output format:**  
                    {{
                        "question": "<Generated question in Swedish>",
                        "answer": "<Generated answer based on the speech in Swedish>"
                    }}
                    """
        
        return self._query_llm(entry, prompt, qa_type)
    
    def generate_qa_comparison_person_yes(self, entry):
        qa_type = "generate_qa_comparison_person"
        prompt = f"""
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant yes/no questions and answers. 
                    
                    Based on the speech below, create a **question in Swedish** that asks about the position of the speaker ({entry['talare']}) as expressed in the speech.
                    Create an **answer in Swedish** based on the question you just created
                    Guidelines:
                    - The answer must always be based on the speech.
                    - Do **not** copy the exact wording from the speech — rephrase using your own words while preserving the intended meaning.
                    - Focus on **broader political opinions or positions**, not on minor or irrelevant details (e.g., do not ask questions like *“Why doesn’t X read the news about herself?”*).
                    - Assume that these are **statements by politicians**, representing **opinions or positions**, not necessarily absolute facts.
                    - Both the question and the answer must be written in **Swedish**, and must be directly relevant to the content of the speech.
                    - The question should expect a yes/no answer, and should start with something like: *"Anser {entry['talare']} att..."*
                    - The answer must always be **"yes"**, followed by a brief **explanation or context** based on the speech.

                    ### Speech ###
                    {entry['anforandetext']}
                    ##################

                    **Output format:**  
                    {{
                        "question": "<Generated question in Swedish>",
                        "answer": "<Generated answer based on the speech in Swedish>"
                    }}
                    """
        
        return self._query_llm(entry, prompt, qa_type)
    
    def generate_qa_comparison_person_no(self, entry):
        qa_type = "generate_qa_comparison_person"
        prompt = f"""
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant yes/no questions and answers. 
                    
                    Based on the speech below, create a **question in Swedish** that asks about the position of the speaker ({entry['talare']}) as expressed in the speech.
                    Create an **answer in Swedish** based on the question you just created
                    Guidelines:
                    - The answer must always be based on the speech.
                    - Do **not** copy the exact wording from the speech — rephrase using your own words while preserving the intended meaning.
                    - Focus on **broader political opinions or positions**, not on minor or irrelevant details (e.g., do not ask questions like *“Why doesn’t X read the news about herself?”*).
                    - Assume that these are **statements by politicians**, representing **opinions or positions**, not necessarily absolute facts.
                    - Both the question and the answer must be written in **Swedish**, and must be directly relevant to the content of the speech.
                    - The question should expect a yes/no answer, and should start with something like: *"Anser {entry['talare']} att..."*
                    - The answer must always be **"no"**, followed by a brief **explanation or context** based on the speech.

                    ### Speech ###
                    {entry['anforandetext']}
                    ##################

                    **Output format:**  
                    {{
                        "question": "<Generated question in Swedish>",
                        "answer": "<Yes-answer with brief explanation in Swedish>"
                    }}
                    """
        
        return self._query_llm(entry, prompt, qa_type)
    


    def generate_qa_comparison_party_yes(self, entry):
        qa_type = "generate_qa_comparison_party"
        prompt = f"""
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant yes/no questions with corresponding answers.

                    Based on the speech below, create a **question in Swedish** that asks about the position of the party ({entry['parti']}) as expressed in the speech.
                    Create an **answer in Swedish** based on the question you just created
                    Guidelines:
                    - Use the **full name of the party**, not the abbreviation (see list below).
                    - Do **not** copy the exact wording from the speech — rephrase using your own words while preserving the intended meaning.
                    - Focus on **broader political opinions or positions**, not on minor or irrelevant details (e.g., do not ask questions like *“Why doesn’t X read the news about herself?”*).
                    - Assume that these are **statements by politicians**, representing **opinions or positions**, not necessarily absolute facts.
                    - Both the question and the answer must be written in **Swedish**, and must be directly relevant to the content of the speech.
                    - The question should expect a yes/no answer, and should start with something like: *"Anser {entry['parti']} att..."*
                    - The answer must always be **"yes"**, followed by a brief **explanation or context** based on the speech.

                    **Party abbreviations and their full names:**
                    M: Moderaterna  
                    S: Socialdemokraterna  
                    SD: Sverigedemokraterna  
                    C: Centerpartiet  
                    V: Vänsterpartiet  
                    L: Liberalerna  
                    KD: Kristdemokraterna  
                    MP: Miljöpartiet

                    ### Speech ###
                    {entry['anforandetext']}
                    ##################

                    **Output format:**  
                    {{
                        "question": "<Generated question in Swedish>",
                        "answer": "<Yes-answer with brief explanation in Swedish>"
                    }}
                    """

        return self._query_llm(entry, prompt, qa_type)
    

    def generate_qa_comparison_party_no(self, entry):
        qa_type = "generate_qa_comparison_party"
        prompt = f"""
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant yes/no questions with corresponding answers.

                    Based on the speech below, create a **question in Swedish** that asks about the position of the party ({entry['parti']}) as expressed in the speech.
                    Create an **answer in Swedish** based on the question you just created
                    Guidelines:
                    - Use the **full name of the party**, not the abbreviation (see list below).
                    - Do **not** copy the exact wording from the speech — rephrase using your own words while preserving the intended meaning.
                    - Focus on **broader political opinions or positions**, not on minor or irrelevant details (e.g., do not ask questions like *“Why doesn’t X read the news about herself?”*).
                    - Assume that these are **statements by politicians**, representing **opinions or positions**, not necessarily absolute facts.
                    - Both the question and the answer must be written in **Swedish**, and must be directly relevant to the content of the speech.
                    - The question should expect a yes/no answer, and should start with something like: *"Anser {entry['parti']} att..."*
                    - The answer must always be **"no"**, followed by a brief **explanation or context** based on the speech.

                    **Party abbreviations and their full names:**
                    M: Moderaterna  
                    S: Socialdemokraterna  
                    SD: Sverigedemokraterna  
                    C: Centerpartiet  
                    V: Vänsterpartiet  
                    L: Liberalerna  
                    KD: Kristdemokraterna  
                    MP: Miljöpartiet

                    ### Speech ###
                    {entry['anforandetext']}
                    ##################

                    **Output format:**  
                    {{
                        "question": "<Generated question in Swedish>",
                        "answer": "<Yes-answer with brief explanation in Swedish>"
                    }}
                    """        
        return self._query_llm(entry, prompt, qa_type)
    
    def generate_qa_temporal(self, entry):
        qa_type = "generate_qa_temporal"
        prompt = f"""
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant questions and answers. 

                    Based on the following speech, create a question that asks about the person's ({entry['talare']}) position based on the speech during a specific time {entry["dok_datum"]}.
                    The questions must include the speaker and the a time perdiod, preferrably a month (e.g., "Vad sade {entry['talare']} om x under {entry["dok_datum"]}?").  
                    Guidelines:
                    - Do **not** copy the exact wording from the speech — rephrase using your own words while preserving the intended meaning.
                    - Focus on **broader political opinions or positions**, not on minor or irrelevant details (e.g., do not ask questions like *“Why doesn’t X read the news about herself?”*).
                    - Assume that these are **statements by politicians**, representing **opinions or positions**, not necessarily absolute facts.
                    - Both the question and the answer must be written in **Swedish**, and must be directly relevant to the content of the speech.
                    - The question and answer must be written in **Swedish** and should be directly relevant to the provided speech.
                    - In  the data, the date is written as numbers, but in your questions, months **must be written out**. But years should be written with numbers.
                    - You should **not** mention the days, only the months.
                    - For example, 2022-03-16 00:00:00 should be written as "i mars 2022"

                    ### Speech ###
                    {entry['anforandetext']}
                    ##################

                    **Output format:**  
                    {{
                        "question": "<Generated question in Swedish>",
                        "answer": "<Generated answer based on the speech in Swedish>"
                    }}
                    """
        
        return self._query_llm(entry, prompt, qa_type)



    def _query_llm(self, entry, prompt, qa_type):
        context = entry.get("anforandetext", "")
        if not context:
            print(f"Skipping entry {entry.get('anforande_id', 'UNKNOWN')} due to missing text.")
            return None

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context:\n{entry['anforandetext']}"}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        try:
            response = requests.post(self.url, json=body, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            message_content = data["choices"][0]["message"]["content"]
            qa_pair = json.loads(message_content)
            return self.format_output(entry, qa_pair, qa_type)
        
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
        except json.JSONDecodeError:
            print(f"Failed to parse response for {entry.get('anforande_id', 'UNKNOWN')}: {data}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        return None
    
    def format_output(self, entry, qa_pair, qa_type):
        return {
            "dok_id": entry.get("dok_id", ""),
            "avsnittsrubrik": entry.get("avsnittsrubrik", ""),
            "dok_datum": entry.get("dok_datum", ""),
            "anforande_id": entry.get("anforande_id", ""),
            "anforandetext": entry.get("anforandetext", ""),
            "talare": entry.get("talare", ""),
            "parti": entry.get("parti", ""),
            "qa_pair": qa_pair,
            "qa_type": qa_type,
            "human_annotator": {
                "reasonable_question": "",
                "context": []
            },
            "LLM_annotator": "",
            "graph_RAG_cosine": {
                        "answer": "",
                        "context": [],
                        "cypher_query": ""},
            "graph_RAG_bm25": {
                        "answer": "",
                        "context": [],
                        "cypher_query": ""},
            "cosine_RAG": {
                        "answer": "",
                        "context": []
            },
            "bm25_RAG": {
                        "answer": "",
                        "context": []                       
            },
            "eval": {
                "graph_RAG_cosine_response": "",
                "graph_RAG_bm25_response": "",
                "cosine_RAG_response": "",
                "bm25_RAG_response": "",
                "orig_answer": ""
            }
            }
        
if __name__ == "__main__":
    input_file = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/more_dataset_random_entries.json"
    output_file = "/mnt/c/Users/User/thesis/data_import/data_small_size/qa_dataset_more_random_entries.json"
    QG = QueryGenerator()

    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    qa_dataset = []
    
    #ratios = {
    #    "generate_qa_inference_person": 0.2,
    #    "generate_qa_inference_party": 0.2,
    #    "generate_qa_comparison_person_yes": 0.12,
    #    "generate_qa_comparison_person_no": 0.12,
    #    "generate_qa_comparison_party_yes": 0.12,
    #    "generate_qa_comparison_party_no": 0.12,
    #    "generate_qa_temporal": 0.12
    #}

    # Number of questions for each category
    required_questions = {
        "generate_qa_inference_person": 36,
        "generate_qa_inference_party": 41,
        "generate_qa_comparison_person_yes": 13, #27
        "generate_qa_comparison_person_no": 12,
        "generate_qa_comparison_party_yes": 16, #32
        "generate_qa_comparison_party_no": 16,
        "generate_qa_temporal": 41
    }

    generated_counts = {
        "generate_qa_inference_person": 0,
        "generate_qa_inference_party": 0,
        "generate_qa_comparison_person_yes": 0,
        "generate_qa_comparison_person_no": 0,
        "generate_qa_comparison_party_yes": 0,
        "generate_qa_comparison_party_no": 0,
        "generate_qa_temporal": 0
    }

    for anforande in data:
        for query_type, required_amount in required_questions.items():
            # Check if we still need to generate more of this question type
            if generated_counts[query_type] < required_amount:
                if query_type == "generate_qa_inference_person":
                    qa_pair = QG.generate_qa_inference_person(anforande)
                elif query_type == "generate_qa_inference_party":
                    qa_pair = QG.generate_qa_inference_party(anforande)
                elif query_type == "generate_qa_comparison_person_yes":
                    qa_pair = QG.generate_qa_comparison_person_yes(anforande)
                elif query_type == "generate_qa_comparison_person_no":
                    qa_pair = QG.generate_qa_comparison_person_no(anforande)
                elif query_type == "generate_qa_comparison_party_yes":
                    qa_pair = QG.generate_qa_comparison_party_yes(anforande)
                elif query_type == "generate_qa_comparison_party_no":
                    qa_pair = QG.generate_qa_comparison_party_no(anforande)
                elif query_type == "generate_qa_temporal":
                    qa_pair = QG.generate_qa_temporal(anforande)

                if qa_pair:
                    print(qa_pair)
                    qa_dataset.append(qa_pair)
                    generated_counts[query_type] += 1
                time.sleep(1)

                # Stop generating for this query type if we have enough
                if generated_counts[query_type] >= required_amount:
                    break


    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(qa_dataset, file, ensure_ascii=False, indent=4)

    print(f"Saved {len(qa_dataset)} Q&A pairs to {output_file}")

    '''for anforande in data:
        random_value = random.random()
        
        # Randomly select the query type based on the specified ratios
        cumulative_probability = 0
        for query_type, ratio in ratios.items():
            cumulative_probability += ratio
            if random_value < cumulative_probability:
                break

        if query_type == "generate_qa_inference_person":
            qa_pair = QG.generate_qa_inference_person(anforande)
        elif query_type == "generate_qa_inference_party":
            qa_pair = QG.generate_qa_inference_party(anforande)
        elif query_type == "generate_qa_comparison_person_yes":
            qa_pair = QG.generate_qa_comparison_person_yes(anforande)
        elif query_type == "generate_qa_comparison_person_no":
            qa_pair = QG.generate_qa_comparison_person_no(anforande)
        elif query_type == "generate_qa_comparison_party_yes":
            qa_pair = QG.generate_qa_comparison_party_yes(anforande)
        elif query_type == "generate_qa_comparison_party_no":
            qa_pair = QG.generate_qa_comparison_party_no(anforande)
        elif query_type == "generate_qa_temporal":
            qa_pair = QG.generate_qa_temporal(anforande)


        if qa_pair:
            qa_dataset.append(qa_pair)'''
        
