import os
import openai
import json
import time

import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
url = "https://api.openai.com/v1/chat/completions"
input_file = "/mnt/c/Users/User/thesis/data_import/exp2/random_sample.json"
output_file = "/mnt/c/Users/User/thesis/data_import/exp2/qa_dataset.json"


class QueryGenerator:
    def __init__(self):
        pass
            
    def generate_qa(self, entry):
        prompt =prompt = f"""
                        You are an expert in analyzing Swedish parliamentary debates and generating relevant questions and answers. 

                        Based on the following speech, create a question that either:  
                        1. Asks about what the speaker ({entry['talare']}) has expressed in the speech (e.g., "Vad säger {entry['talare']} om...?" or "Hur argumenterar {entry['talare']} för...?)".  
                        2. Asks about the party's ({entry['parti']}) position based on the speech (e.g., "Vad är {entry['parti']}:s position i...?").  

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

        body = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Context:\n{entry['anforandetext']}"}
                ],
                "max_tokens": 500,
                "temperature": 0
            }
        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

        context = entry.get("anforandetext", "")

        if not context:
            print(f"Skipping entry {entry.get('anforande_id', 'UNKNOWN')} due to missing text.")
            return None
        try:
            response = requests.post(url, json=body, headers=headers)
            response.raise_for_status()  # Raise an error if the request fails

            data = response.json()
            message_content = data["choices"][0]["message"]["content"]
            qa_pair = json.loads(message_content)

            # Include the additional fields in the output
            enriched_entry = {
                "avsnittsrubrik": entry.get("avsnittsrubrik", ""),
                "anforande_id": entry.get("anforande_id", ""),
                "anforandetext": entry.get("anforandetext", ""),
                "talare": entry.get("talare", ""),
                "parti": entry.get("parti", ""),
                "qa_pair": qa_pair  # The generated question and answer
            }

            return enriched_entry

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
        except json.JSONDecodeError:
            print(f"Failed to parse response for {entry.get('anforande_id', 'UNKNOWN')}: {data}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        return None


if __name__ == "__main__":
    QG = QueryGenerator()

    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    qa_dataset = [] 
    for anforande in data:
        qa_pair = QG.generate_qa(anforande)
        if qa_pair:
            qa_dataset.append(qa_pair)

        time.sleep(1)  

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(qa_dataset, file, ensure_ascii=False, indent=4)

    print(f"Saved {len(qa_dataset)} Q&A pairs to {output_file}")