import requests
import json
import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
N_GENERATIONS = 5  # Generate only 5 QA pairs for cost and time considerations
dataset = load_dataset("json", data_files="/mnt/c/Users/User/thesis/data_import/exp1/even_more_filtered_riksdag.json", split="train")
#dataset = dataset.filter(lambda x: x["dok_id"] in ["H90968", "H90982"])
url = "https://api.openai.com/v1/chat/completions"


system_prompt = """
                Your task is to generate factoid question-answer pairs in **Swedish**, based on **spoken statements ("anforandetext") from debates in the Swedish parliament (Riksdagen).** 

                ### **Important Instructions**
                - Each **factoid question** must:
                - Be written in **natural Swedish**, as if a user were searching online.
                - Be **factually correct** and based on actual statements made in the debate.
                - **Not mention metadata fields** like "systemdatum", "anforande_id", or dataset structure.
                - **Focus on arguments and discussions** from members of parliament.
                - Be **specific** and answerable with a **concise, factual response**.

                - Each **answer** must:
                - Provide a **concise and factual** response based on the debate text.
                - Be **directly supported** by a quoted excerpt from "anforandetext."

                ---

                ### **Output Format**
                Each factoid QA pair must follow this exact structure:

                Factoid question: (Your generated factoid question)
                Answer: (Your factually correct answer based on the context)
                Context: (The "anforande_id" that supports your answer)

                Examples:
                Factoid question: Hur argumenterar Anders Borg om restaurangmomsen?
                Answer: Anders Borg menar att sänkt restaurangmoms leder till fler jobb inom restaurangbranschen.
                Context: "59a228e8-3dc6-ec11-9170-0090facf175a"

                Factoid question: Hur debatterar Vänsterpartiet om vinster i välfärden?
                Answer: Vänsterpartiet argumenterar för att vinster i välfärden bör begränsas för att säkerställa att skattepengar går till omsorg och utbildning, inte till privata aktörer.
                Context: "5aa228e8-3dc6-ec11-9170-0090facf175a"

                ### **🚫 Avoid These Mistakes**
                ❌ **DO NOT generate questions about dataset structure** (e.g., "Vad betyder anforande_nummer?")  
                ❌ **DO NOT write any additional text except what is instructed above** (e.g., "Här kommer frågor om...")  
                ❌ **DO NOT create vague or opinion-based questions** (e.g., "Vad tycker folk om skatter?")  
                """
def generate_qa_from_openai(context, anforande_id, bonus_prompt):
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": bonus_prompt+system_prompt},
            {"role": "user", "content": f"Context:\n{context}"}
        ],
        "max_tokens": 500,
        "temperature": 0
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    response = requests.post(url, json=body, headers=headers)

    if response.status_code == 200:
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]

        # Split the response into lines
        qa_lines = response_text.strip().split("\n")

        # Check if there are at least two lines for a question and an answer
        if len(qa_lines) >= 2:
            question_line = qa_lines[0].replace("Factoid question:", "").strip()
            answer_line = qa_lines[1].replace("Answer:", "").strip()

            # Return a single question-answer pair
            return {"question": question_line, "answer": answer_line, "source_doc": anforande_id}
        else:
            return None
    else:
        print("Error:", response.text)
        return None

qa_results = list()
for example in dataset:
    anforandetext = example["anforandetext"]
    anforande_id = example["anforande_id"]
    theme = example["avsnittsrubrik"]
    speaker = example["talare"]
    party = example["parti"]

    prompt_addon = f"""
                    Generate a question-answer pair based on this statement by {speaker} from the party {party}
                    From the debate about {theme}.

                    Start the question like this:
                    "Vad säger {speaker} om..." or "Hur argumenterar {speaker} för..."

                    And always start the response by mentioning {speaker}'s name.

                    ALWAYS write both a question and an answer!!
                    """
    
    qa_pair = generate_qa_from_openai(anforandetext, anforande_id, prompt_addon)

    if qa_pair:
        qa_results.append({
            "anforande_id": example["anforande_id"],
            "dok_id": example["dok_id"],
            "talare": example["talare"],
            "question": qa_pair["question"],
            "answer": qa_pair["answer"],
            "context": anforandetext
        })

print(json.dumps(qa_results, ensure_ascii=False, indent=2))
'''


Use for future prompting for getting other kinds of questions
                Comparison questions or yes/no questions
                For example: "Tycker Carl Bildt att arbetsgivaravgiten ska höjas?"
                Temporal or time related questions:
                For example: "Vad argumenterade Jonas Sjöstedt för i maj 2018?"
                And null questions, meaning questions that cannot be answered given the database:
                For example: "Vad tycker Johan Pehrsson om vargjakt?"

previous_history = [
    {"role": "user", "message": """Generate a question-answer pair in Swedish based on what Anders Borg says in the context below.

    Context:
    Anders Borg: \r\nFru talman! Utgångspunkten för den här debatten är frågan om regeringens ekonomiska politik har fungerat.\r\nVärlden befinner sig i den kanske djupaste och längsta recession som vi har upplevt sedan depressionen. Land efter land har fallit ned i en ekonomisk abyss.\r\nSverige står ut jämfört med nästan alla industriländer när man tittar samlat på utvecklingen. Det är en bättre tillväxt, högre reallöner, starkare konsumtionsutveckling, hög sysselsättningsgrad och låg långtidsarbetslöshet. Det är inkomstskillnader som inte har ökat som de har gjort i andra länder och en minskad andel som lever med låg materiell standard, tack vare att vi har kunnat höja konsumtionen i Sverige, trots att det har varit en kraftig lågkonjunktur.\r\nDet här beror naturligtvis på att vi har fört en samlad ekonomisk politik som präglas av ansvar, arbete och kunskap. Jobbskatteavdrag och förändringar i socialförsäkringen och förändringar av a-kassa har gjort att det lönar sig bättre att arbeta.\r\nNystartsjobb och förändringar av arbetsmarknadspolitiken har gjort att vi stöttar dem som har svårigheter på arbetsmarknaden. Omläggning av utbildningspolitiken i riktning mot mer av kunskap, betyg och prov gör att vi lägger en starkare grund. Bättre företagsklimat med borttagen arvsskatt och förmögenhetsskatt och sänkt bolagsskatt gör att Sverige står sig väl.\r\nNu hävdar debattörerna här att vi inte har haft några forskare som sagt att regeringens politik kan ha bidragit till detta. Jag vill bara påpeka att KI:s bedömning är att någonstans runt 1 procent i lägre långsiktig arbetslöshet är effekten av regeringens politik.\r\nRiksbanken säger att det är ½ till 1 ½ procent eller 1 ½ till 2 procent, om jag minns siffrorna rätt. Finansdepartementet landar på siffror som ligger väldigt nära Riksbankens och KI:s. Vi har OECD och IMF och Finanspolitiska rådet som säger exakt samma sak.\r\nSocialdemokraterna försöker driva en tes, det vill säga att marginaleffekten, värdet av att arbeta när man tar hänsyn till skatter och bidrag, inte ska påverka arbetsutbudet. Det är Socialdemokraternas tes.\r\nDet finns helt enkelt inga forskare som stöder den slutsatsen, utan forskningen är klar och tydlig. Hur mycket det lönar sig att arbeta och hur väl lönebildningen fungerar är det som avgör hur stort arbetsutbud vi får. Och det är det som bestämmer den långsiktiga sysselsättningsgraden.\r\nJag måste säga att jag är tacksam för att Socialdemokraterna gör att vi kommer att få en tydlig och bra valrörelse.\r\nSocialdemokraterna förordar en politik där det ena benet är stora skattehöjningar. Höjda arbetsgivaravgifter för unga, höjd restaurangmoms och höjd bolagsskatt är hörnpelarna. Det ska användas till en bred utbyggnad av bidragssystemen. Praktiskt taget varje bidragssystem ska byggas ut. Det som tillkommit under den här senaste budgetomgången är barnbidraget. Men det är också a-kassan. Det är sjukförsäkringen. Förtidspensioneringen ska öppnas och så vidare.\r\nI en välfärdsstat med en åldrande befolkning har alltså Socialdemokraterna det relativt unika receptet att man ska höja skatterna på arbete och efterfrågan för att kraftigt bygga ut socialförsäkringssystemet så att fler människor lämnar arbetsmarknaden. Det är en väldigt ovanlig ekonomisk politik.\r\nVad värre är är att Socialdemokraterna tänker genomföra den här politiken med Miljöpartiet, och Miljöpartiet har en egen idé om ekonomisk politik. Det är en stor satsning på miljöinvesteringar som ska finansieras med kraftigt höjda bensin- och transportskatter. Det är Miljöpartiets politik.\r\nLägger man ihop den här bilden blir det ännu märkligare. Först ska vi skära ned efterfrågan med höjd moms, höjd bensinskatt och höjda transportskatter. Sedan ska vi lägga på kostnader i form av höjda arbetsgivaravgifter, höjd bolagsskatt och försämringar av företagsklimatet. Det är klart att den åtstramningspolitiken kommer att leda till att färre människor har arbete.\r\nNär man sedan lägger på att denna politik används i syfte att bygga ut bidragssystemen så att fler människor ska lämna arbetsmarknaden kan man ju inte få någon annan effekt än att om man gör det dyrare och mindre lönsamt att arbeta, då lämnar människor arbetsmarknaden, och då stiger arbetslösheten och faller sysselsättningen.\r\n"""
    }
]

payload = {
    "providers": "deepseek/DeepSeek-V3", #"meta/llama3-1-405b-instruct-v1:0",
    "response_as_dict": True,
    "attributes_as_list": False,
    "show_base_64": True,
    "show_original_response": False,
    "temperature": 0,
    "max_tokens": 4096,
    "tool_choice": "auto",
    "previous_history": previous_history,
    "chatbot_global_action": global_message
}

headers = {
    "Authorization": f"Bearer {EDENAI_API_KEY}",
    "accept": "application/json",
    "content-type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)



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

print(outputs)'''
