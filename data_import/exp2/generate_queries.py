import os
import random
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

                    Based on the following speech, asks about what the speaker ({entry['talare']}) has expressed in the speech (e.g., "Vad säger {entry['talare']} om...?" or "Hur argumenterar {entry['talare']} för...?)".  

                    The question and answer must be written in **Swedish** and should be directly relevant to the provided speech.

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

                    Based on the following speech, create a question that asks about the party's ({entry['parti']}) position based on the speech (e.g., "Vad är {entry['parti']}:s position i...?").  

                    The question and answer must be written in **Swedish** and should be directly relevant to the provided speech.
                    In the data, the party association is written as the short form, but in your questions, the names **must be written out**.
                    M: Moderaterna, S: Socialdemokraterna, SD: Sverigedemokraterna, C: Centerpartiet, V: Vänsterpartiet, L: Liberalerna, KD: Kristdemokraterna, MP: Miljöpartiet de Gröna

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
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant questions and answers. 

                    Based on the following speech, create a question that asks about the person's ({entry['talare']}) position based on the speech.
                    The must be possible to answer with a yes/no answer (e.g., "Anser {entry['talare']} att...?").  
                    When writing responses to the questions, provide a yes/no answer as well as additional information.
                    You must write a question to which the answer is **yes**.

                    
                    The question and answer must be written in **Swedish** and should be directly relevant to the provided speech.

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
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant questions and answers. 

                    Based on the following speech, create a question that asks about the person's ({entry['talare']}) position based on the speech.
                    The must be possible to answer with a yes/no answer (e.g., "Anser {entry['talare']} att...?").  
                    When writing responses to the questions, provide a yes/no answer as well as additional information.
                    You must write a question to which the answer is **no**.
                    Do **not** use double negations like "Anser x att y inte är z?"

                    
                    The question and answer must be written in **Swedish** and should be directly relevant to the provided speech.

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
    


    def generate_qa_comparison_party_yes(self, entry):
        qa_type = "generate_qa_comparison_party"
        prompt = f"""
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant questions and answers. 

                    Based on the following speech, create a question that asks about the person's ({entry['parti']}) position based on the speech.
                    The must be possible to answer with a yes/no answer (e.g., "Anser {entry['parti']} att...?").  
                    When writing responses to the questions, provide a yes/no answer as well as additional information.
                    You must write a question to which the answer is **yes**.


                    The question and answer must be written in **Swedish** and should be directly relevant to the provided speech.
                    In the data, the party association is written as the short form, but in your questions, the names **must be written out**.
                    M: Moderaterna, S: Socialdemokraterna, SD: Sverigedemokraterna, C: Centerpartiet, V: Vänsterpartiet, L: Liberalerna, KD: Kristdemokraterna, MP: Miljöpartiet de Gröna

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
    

    def generate_qa_comparison_party_no(self, entry):
        qa_type = "generate_qa_comparison_party"
        prompt = f"""
                    You are an expert in analyzing Swedish parliamentary debates and generating relevant questions and answers. 

                    Based on the following speech, create a question that asks about the person's ({entry['parti']}) position based on the speech.
                    The must be possible to answer with a yes/no answer (e.g., "Anser {entry['parti']} att...?").  
                    When writing responses to the questions, provide a yes/no answer as well as additional information.
                    You must write a question to which the answer is **no**.
                    Do **not** use double negations like "Anser x att y inte är z?"

                    The question and answer must be written in **Swedish** and should be directly relevant to the provided speech.
                    In the data, the party association is written as the short form, but in your questions, the names **must be written out**.
                    M: Moderaterna, S: Socialdemokraterna, SD: Sverigedemokraterna, C: Centerpartiet, V: Vänsterpartiet, L: Liberalerna, KD: Kristdemokraterna, MP: Miljöpartiet de Gröna

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
            "avsnittsrubrik": entry.get("avsnittsrubrik", ""),
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
            "LLM_annotator": {
                "reasonable_question": ""
            }
        }


if __name__ == "__main__":
    input_file = "/mnt/c/Users/User/thesis/data_import/exp2/random_sample.json"
    output_file = "/mnt/c/Users/User/thesis/data_import/exp2/qa_dataset.json"
    QG = QueryGenerator()

    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    qa_dataset = []
    
    ratios = {
        "generate_qa_inference_person": 0.2,
        "generate_qa_inference_party": 0.2,
        "generate_qa_comparison_person_yes": 0.15,
        "generate_qa_comparison_person_no": 0.15,
        "generate_qa_comparison_party_yes": 0.15,
        "generate_qa_comparison_party_no": 0.15
    }

    for anforande in data:
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


        if qa_pair:
            qa_dataset.append(qa_pair)
        
        time.sleep(1)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(qa_dataset, file, ensure_ascii=False, indent=4)

    print(f"Saved {len(qa_dataset)} Q&A pairs to {output_file}")