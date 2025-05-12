import json
import time
from datasets import load_dataset
import os
from dotenv import load_dotenv
import requests

load_dotenv()
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")
#dataset = load_dataset("json", data_files="/mnt/c/Users/User/thesis/data_import/data_small_size/data/evaluated_dataset.json", split="train")
url = "https://api.edenai.run/v2/text/chat"


global_message = """
You are an evaluator tasked with judging a questions and responses. 
Your job is to evaluate if the **response** answers the **question**.
---

### Return Yes if:
✅ If the response directly answers the question.
---
### Return No if: 
❌ The response says it cannot answer the question.
❌ The response strays from the question.
---

Return your final evaluation using **only** the following format:
Yes/No

Do not include any other text, explanation, or formatting.
"""


headers = {
    "Authorization": f"Bearer {EDENAI_API_KEY}",
    "accept": "application/json",
    "content-type": "application/json"
}

def evaluate_rag_output_with_edenai(question, answer, context):
    payload = {
        "providers": "google/gemini-1.5-pro-latest", #"openai/gpt-4o",  #"meta/llama3-1-405b-instruct-v1:0",     #"deepseek/DeepSeek-V3",
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_base_64": True,
        "show_original_response": False,
        "temperature": 0,
        "max_tokens": 4096,
        "tool_choice": "auto",
        "previous_history": [
            {'role': 'user', 'message': f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"}, 
            {'role': 'user', 'message': global_message}
        ]
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        evaluation = response.json()
        #print(json.dumps(evaluation, indent=2))  # <-- Add this line to print full LLM response nicely
        return evaluation

    else:
        print("Error with EdenAI API request:", response.text)
        return None

def get_generated_text_safely(evaluation, tag=""):
    try:
        response = evaluation.get("google/gemini-1.5-pro-latest", {})
        if response.get("status") != "success":
            print(f"[{tag}] Response status not successful:", response.get("status"))
            print(json.dumps(response, indent=2))
            return None
        return response["generated_text"]
    except Exception as e:
        print(f"[{tag}] Error accessing 'generated_text': {e}")
        print("Full response for debugging:")
        print(json.dumps(evaluation, indent=2))
        return None

dataset = [{
    "dok_id": "H809118",
    "avsnittsrubrik": "Svar på interpellation 2020/21:665 om åtgärder för att möjliggöra kulturupplevelser utomhus med minskad risk för smittspridning",
    "dok_datum": "2021-05-04 00:00:00",
    "anforande_id": "be0c5c85-3a0b-ec11-9160-901b0e9b71a0",
    "anforandetext": "<p>Fru talman! Jag tackar Gudrun Brunegård för denna fråga. Jag vill börja med att säga att jag verkligen delar den oro som riksdagsledamoten uttrycker för dessa verksamheter och är också mån om verksamheterna. Om man tar Astrid Lindgrens Värld som exempel är det helt klart ett besöksmål som har en enorm betydelse för lokalsamhället. Och jag vet att det var en väldigt tuff sommar förra sommaren. </p><p>Jag ska börja med restriktionerna. Det har varit en ganska trixig resa från att vi hade den gamla ordningslagen som gjorde att det slog väldigt olyckligt förra sommaren. Jag tror att vi alla är överens om att det inte var verksamheternas förutsättningar att hålla avstånd som det handlade om utan vilka verksamheter som då fanns med i regleringen i ordningslagen. </p><p>Sedan har pandemin tyvärr också fortsatt skölja över vårt land. Vi har haft en andra våg och en tredje våg, vilket har gjort att det träffsäkra regelverket ännu inte har kunnat komma på plats. Vi har nästan dagligen kontakt med Folkhälsomyndigheten från regeringens sida så att vi så snabbt som möjligt ska kunna ta detta beslut. </p><p>Det som har hänt sedan jag lämnade detta svar till Gudrun Brunegård är att regeringen förra veckan presenterade ett uppdaterat förslag på detta, där man har lyssnat in remissynpunkterna och tagit fram en ytterligare bestämmelse just för sittande publik utomhus. Där finns det möjlighet i enlighet med det första steget att ta emot upp till 500 personer. </p><p>Jag vill också understryka att tanken med detta är att Folkhälsomyndigheten ska komplettera dessa deltagartak med föreskrifter som till exempel skulle kunna handla om avstånd mellan sittplatser och hur många man får sitta i sällskap. Det innebär de facto att en arenas eller en lokals beskaffenhet eller yta kommer att kunna påverka hur många man får ta in eftersom avståndskrav måste gälla. </p><p>Sedan vill jag svara på Gudrun Brunegårds fråga. Det är klart att det finns väldigt många anläggningar som steg för steg kommer att kunna ta emot mer publik. Vi går förhoppningsvis åt det hållet nu i och med vaccinet, och jag tror att vi alla ser framför oss en sådan utveckling. Men detta är verkligen att betrakta som ett första steg och i en situation då det fortfarande är en väldigt stor smittspridning. </p><p>Folkhälsomyndigheten har lämnat in förslag på vilka deltagartak som skulle kunna gälla vid tre olika nivåer på smittspridning. Och detta är alltså vad som skulle kunna gälla på den högsta nivån av smittspridning enligt Folkhälsomyndighetens förslag. Sedan ser vi nog alla framför oss hur vi steg för steg ska kunna ta ytterligare beslut för att öppna för mer och mer publik, i takt med vad smittläget tillåter.</p><p> STYLEREF Kantrubrik \\* MERGEFORMAT Svar på interpellationer</p><p>Jag vill också kommentera det här med vad Astrid Lindgrens Värld är. Det är ju en kombination av flera olika verksamheter. Det som kommer att hända när beslutet om de mer träffsäkra reglerna väl är taget är att restauranger i Astrid Lindgrens Värld kommer att regleras enligt restauranglagen. Teaterverksamheter kommer att falla inom ramen för publikreglerna. Men själva evenemangsparken kommer att lyda under samma regler som andra nöjesparker. Då kommer vi ifrån den olikhet som rådde förra året - att det handlade om hur man bedömer vilken typ av park det är. Som jag har uppfattat det som kommer att börja gälla kommer det att handla om olika regelverk för de olika delarna. Det tror jag blir bra - då hamnar vi i en mer logisk situation, och det blir inte för stora skillnader mellan olika delar av verksamheterna.</p>",
    "talare": "Kultur- och demokratiministern Amanda Lind",
    "parti": "MP",
    "qa_pair": {
      "question": "Anser Kultur- och demokratiministern Amanda Lind att de nuvarande restriktionerna är tillräckliga för att hantera smittspridningen och samtidigt stödja verksamheter som Astrid Lindgrens Värld?",
      "answer": "Nej, Kultur- och demokratiministern Amanda Lind anser inte att de nuvarande restriktionerna är tillräckliga. Hon betonar att det fortfarande finns en stor smittspridning och att de nuvarande åtgärderna är ett första steg. Hon ser fram emot att stegvis kunna öppna upp mer i takt med att smittläget förbättras och att mer träffsäkra regler kan införas."
    },
    "qa_type": "generate_qa_comparison_person",
    "human_annotator": {
      "reasonable_question": "",
      "context": []
    },
    "LLM_annotator": "",
    "graph_RAG_cosine": {
      "answer": "Kultur- och demokratiminister Amanda Lind verkar tydligt medveten om svårigheterna för verksamheter som Astrid Lindgrens Värld under pandemin. Hon uttrycker att de nuvarande restriktionerna har varit svåra att navigera för många verksamheter, men att regeringen arbetar för att skapa en mer anpassningsbar lösning. De nya förslagen om restriktioner presenterades nyligen och innebär att platser som Astrid Lindgrens Värld kan ta emot upp till 500 personer utomhus, förutsatt att vissa avståndskrav följs. Dessutom kommer olika delar av verksamheten, som restauranger och teatrar, att regleras enligt sina respektive lagar.\n\nFör att kunna öppna upp mer för publiken, kommer Folkhälsomyndigheten att komplettera dessa deltagartak med föreskrifter som rör avstånd mellan sittplatser och hur många man får sitta i sällskap. Dessa förändringar är dock bara ett första steg och Lind betonar att det fortfarande råder stor smittspridning och att situationen är skör. Samtidigt verkar Lind vara optimistisk om att man gradvis kommer att kunna ta ytterligare beslut för att öppna för mer och mer publik, i takt med vad smittläget tillåter.\n\nSammanfattningsvis, verkar Amanda Lind anse att de nuvarande restriktionerna, tillsammans med de kommande anpassningarna, är tillräckliga för att hantera smittspridningen samtidigt som de stödjer verksamheter som Astrid Lindgrens Värld. Men hon betonar att situationen är föränderlig och att det krävs försiktighet och anpassning baserat på smittläget.",
      "context": [
        "be0c5c85-3a0b-ec11-9160-901b0e9b71a0",
        "eb9e5c80-3987-eb11-914b-901b0eac4c70",
        "a6c171c9-b822-ec11-915b-901b0eac4c78",
        "9ec171c9-b822-ec11-915b-901b0eac4c78",
        "f79e5c80-3987-eb11-914b-901b0eac4c70",
        "c20c5c85-3a0b-ec11-9160-901b0e9b71a0"
      ],
      "cypher_query": "MATCH (t:Talare {name: \"Kultur- och demokratiministern Amanda Lind\"}) \nMATCH (t)-[:HALLER]->(a:Anforande) \nMATCH (a)-[:HAS_CHUNK]->(c:Chunk) \nMATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll) \nRETURN DISTINCT a.anforande_text, c.text, c.chunk_id, c.embedding, a.anforande_id",
      "nr_of_nodes": 187
    },
    "graph_RAG_bm25": {
      "answer": "Kultur- och demokratiministern Amanda Lind har uttalat sig om de nuvarande restriktionerna och hur de påverkar verksamheter som Astrid Lindgrens Värld. Hon har erkänt att situationen är mycket allvarlig och att smittläget fortfarande är kritiskt. Men hon har också betonat vikten av att verksamheter ska kunna bedrivas smittsäkert, med arrangörer som har expertis inom detta område.\n\nFör att stödja dessa verksamheter har regeringen infört nya regler som kommer att tillåta vissa nöjesparker, som Gröna Lund och Astrid Lindgrens Värld, att ha öppet under vissa förhållanden. Dessa regler fokuserar på antalet personer per kvadratmeter inom nöjesparken.\n\nMinistern underströk dock att denna första fas av återöppnande inte kommer att vara ett stort steg, utan snarare kommer att innebära begränsade möjligheter. Hon betonade att det återstående arbetet kommer att vara inriktat på att kontrollera smittspridningen och att fortsätta vaccineringen. \n\nAmanda Lind nämnde också regeringens långsiktiga arbete med återhämtningsstrategier för kultursektorn, inklusive en återstartsutredning för kulturen och stöd till kulturskapare. Hon påpekar att det kommer att vara nödvändigt att stötta kulturen under en längre tid framåt. \n\nSå, för att svara på din fråga, Amanda Lind anser att de nuvarande restriktionerna är tillräckliga för att hantera smittspridningen, men hon är också medveten om att det behövs mer stöd för att säkerställa att verksamheter som Astrid Lindgrens Värld kan fortsätta att fungera och bidra till kultursektorn.",
      "context": [
        "c00c5c85-3a0b-ec11-9160-901b0e9b71a0",
        "be0c5c85-3a0b-ec11-9160-901b0e9b71a0",
        "a6c171c9-b822-ec11-915b-901b0eac4c78",
        "366efcf1-df3e-eb11-914a-901b0e9b71a8",
        "e25a25c4-37a4-eb11-914f-901b0eac4c78",
        "498eccab-a0ad-eb11-914f-901b0eac4c78"
      ],
      "cypher_query": "MATCH (t:Talare {name: \"Kultur- och demokratiministern Amanda Lind\"}) \nMATCH (t)-[:HALLER]->(a:Anforande) \nMATCH (a)-[:HAS_CHUNK]->(c:Chunk) \nMATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll) \nRETURN DISTINCT a.anforande_text, c.text, c.chunk_id, c.embedding, a.anforande_id",
      "nr_of_nodes": 187
    },
    "cosine_RAG": {
      "answer": "Amanda Lind har uttryckt att det finns en förståelse för de utmaningar som verksamheter som Astrid Lindgrens Värld står inför. Hon har nämnt vikten av att ha träffsäkra regler som tar hänsyn till olika verksamheters förutsättningar att hålla avstånd och hantera smittspridning. Det har varit en komplex process att införa regelverk som både skyddar folkhälsan och stödjer kulturverksamheter. Regeringen arbetar nära Folkhälsomyndigheten för att anpassa restriktionerna i takt med smittläget och har även implementerat ekonomiska stödåtgärder för att underlätta för kultursektorn att återhämta sig. Det finns en ambition att stegvis öppna upp mer, i takt med att smittläget förbättras.",
      "context": [
        "eb9e5c80-3987-eb11-914b-901b0eac4c70",
        "4448d815-7706-ec11-9160-901b0e9b71a0",
        "be0c5c85-3a0b-ec11-9160-901b0e9b71a0",
        "c20c5c85-3a0b-ec11-9160-901b0e9b71a0",
        "f79e5c80-3987-eb11-914b-901b0eac4c70",
        "8075c7f0-3216-ec11-9160-901b0e9b71a0"
      ],
      "scores": [
        0.9127660989761353,
        0.9098432064056396,
        0.9088687300682068,
        0.9061256647109985,
        0.9043760895729065,
        0.9029874801635742
      ]
    }
  }  
]  


updated_data = list()

for idx, entry in enumerate(dataset):
    # Make sure eval_openai exists
    #if "eval_openai" not in entry:
    #    entry["eval_openai"] = {}

    # First original qa-pair
    question = entry.get("qa_pair", {}).get("question", "N/A")
    context = entry.get("anforandetext", "N/A")
    answer = entry.get("qa_pair", {}).get("answer", "N/A")
    print("original: ",question) #, "context: ", context,"answer: ", answer)
    
    # Then, the generated RAG_stuff
    graph_rag_cosine_answer = entry.get("graph_RAG_cosine", {}).get("answer", "N/A")
    graph_rag_bm25_answer = entry.get("graph_RAG_bm25", {}).get("answer", "N/A")
    cosine_rag_answer = entry.get("cosine_RAG", {}).get("answer", "N/A")
    print("ragragrag cosine", graph_rag_cosine_answer)    
    print("ragragrag bm25", graph_rag_bm25_answer)
    print("cosinecosinecosine", cosine_rag_answer)
    print("\n","="*80)

    # Evaluate and insert ONLY missing fields
    #if "graph_RAG_cosine_response" not in entry["eval_openai"]:
    if idx >= 20:
        updated_data.append(entry)
        continue
    else: 
        evaluation_graph_cosine = evaluate_rag_output_with_edenai(question, graph_rag_cosine_answer, context)
        if evaluation_graph_cosine and "google/gemini-1.5-pro-latest" in evaluation_graph_cosine:
            verdict = get_generated_text_safely(evaluation_graph_cosine, tag="graph_RAG_cosine")
            if verdict:
                #entry["gemini_eval"]["graph_RAG_cosine_response"] = verdict
                print("evaluation graph cosine: ", verdict)

    #if "graph_RAG_bm25_response" not in entry["eval_openai"]:
        evaluation_graph_bm25 = evaluate_rag_output_with_edenai(question, graph_rag_bm25_answer, context)
        if evaluation_graph_bm25 and "google/gemini-1.5-pro-latest" in evaluation_graph_bm25:
            verdict = get_generated_text_safely(evaluation_graph_bm25, tag="graph_RAG_bm25")
            if verdict:
                print(entry)
                #entry["gemini_eval"]["graph_RAG_bm25_response"] = verdict
                print("evaluation graph bm25: ", verdict)

    #if "cosine_RAG_response" not in entry["eval_openai"]:
        evaluation_cosine_rag = evaluate_rag_output_with_edenai(question, cosine_rag_answer, context)
        if evaluation_cosine_rag and "google/gemini-1.5-pro-latest" in evaluation_cosine_rag:
            verdict = get_generated_text_safely(evaluation_cosine_rag, tag="cosine_RAG")
            if verdict:
                #entry["gemini_eval"]["cosine_RAG_response"] = verdict
                print("evaluation cosine rag: ", verdict)

    
    updated_data.append(entry)
    time.sleep(0.5)
    print("="*80)

#    print(entry)

#with open("/mnt/c/Users/User/thesis/data_import/data_small_size/data/evaluated_dataset.json", "w", encoding="utf-8") as f:
#    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("Dataset successfully updated and saved!")





old_global_message = """
You are an evaluator tasked with judging a question-answer pair based on two distinct criteria: **Factuality** and **Relevance**. 
Your goal is to assess whether the *question* is grounded in the context, and whether the *answer* appropriately responds to the question.

---

### 1. Factuality: Is the answer based on the context?

✅ Yes, if the answer is clearly grounded in information found in the context.  
❌ No, if the answer introduces topics or facts not supported by the context, or if it is unanswerable based on the context.

---

### 2. Relevance: Does the answer respond to the question?

✅ Yes, if the answer directly and specifically addresses the question.  
❌ No, if the answer ignores the question, is vague, off-topic, or introduces irrelevant or contradictory information.

---

### If the answer states that it cannot answer the question, it should be evaluated as follows:
- **Factuality**: ❌ No, because the question is not grounded in the context.
- **Relevance**: ❌ No, because the answer does not respond to the question.

### If unsure, lean toward "No" for either category unless all conditions for "Yes" are clearly met.

---

### Output Format:
Return your final evaluation as a JSON object using **only** the following format:

{ "factuality": "Yes" or "No", "relevance": "Yes" or "No" }

Do not include any other text, explanation, or formatting.
"""