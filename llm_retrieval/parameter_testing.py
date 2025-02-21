import json
import os
import argparse
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

class CypherModelTester:
    
    def __init__(self):
        self.graph = Neo4jGraph()
        self.graph.refresh_schema()
        self.enhanced_graph = Neo4jGraph(enhanced_schema=True)

    def test_model(self, prompt, model="gpt-4o", temperature=0, system_message=""):
        llm = ChatOpenAI(model=model, temperature=temperature, openai_api_key=OPEN_API_KEY)
        chain = GraphCypherQAChain.from_llm(
            graph=self.enhanced_graph, llm=llm, verbose=False, allow_dangerous_requests=True,
            return_intermediate_steps=True
        )
        messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]
        response = chain.invoke({"query": prompt, "messages": messages})
        return response

    def test_different_models(self, prompt, models, temperature, system_message):
        results = []
        for model in models:
            print(f"Testing model: {model}...")
            response = self.test_model(prompt, model, temperature, system_message)
            results.append({"model": model, "response": response})
        return results

    def test_different_temperatures(self, prompt, model, temperatures, system_message):
        results = []
        for temp in temperatures:
            print(f"Testing temperature: {temp}...")
            response = self.test_model(prompt, model, temp, system_message)
            results.append({"temperature": temp, "response": response})
        return results

    def test_different_system_messages(self, prompt, model, temperature, system_messages):
        results = []
        for sys_msg in system_messages:
            print(f"Testing system message: {sys_msg[:30]}...")  # Print only first 30 chars
            response = self.test_model(prompt, model, temperature, sys_msg)
            results.append({"system_message": sys_msg, "response": response})
        return results

    def save_results(self, filename, model_results, temp_results, sys_msg_results):
        results = {
            "model_tests": model_results,
            "temperature_tests": temp_results,
            "system_message_tests": sys_msg_results
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"All tests completed. Results saved to {filename}.")

    def print_results(self, model_results, temp_results, sys_msg_results):
        print("Model Test Results:")
        for result in model_results:
            print(f"\nModel: {result['model']}")
            print(f"Response: {result['response']['result']}")
            
            print("\nCypher Query:")
            print(result['response']['intermediate_steps'][0]['query'].replace("\\n", "\n"))
            
            print("\nContext Data:")
            for context_item in result['response']['intermediate_steps'][1]['context']:
                print(f"Party: {context_item.get('partyName', 'Unknown')} - Participants: {context_item.get('numParticipants', 'Unknown')}")
            
            print("-" * 50)  
        
        print("\nTemperature Test Results:")
        for result in temp_results:
            print(f"\nTemperature: {result['temperature']}")
            print(f"Response: {result['response']['result']}")
            
            print("\nCypher Query:")
            print(result['response']['intermediate_steps'][0]['query'].replace("\\n", "\n"))
            
            print("\nContext Data:")
            for context_item in result['response']['intermediate_steps'][1]['context']:
                print(f"Party: {context_item.get('partyName', 'Unknown')} - Participants: {context_item.get('numParticipants', 'Unknown')}")
            
            print("-" * 50)

        print("\nSystem Message Test Results:")
        for result in sys_msg_results:
            print(f"\nSystem Message: {result['system_message'][:30]}...")  # Print first 30 chars of the system message
            print(f"Response: {result['response']['result']}")
            
            print("\nCypher Query:")
            print(result['response']['intermediate_steps'][0]['query'].replace("\\n", "\n"))
            
            print("\nContext Data:")
            for context_item in result['response']['intermediate_steps'][1]['context']:
                print(f"Party: {context_item.get('partyName', 'Unknown')} - Participants: {context_item.get('numParticipants', 'Unknown')}")
            
            print("-" * 50) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test different prompts, models, and parameters.")
    parser.add_argument("--prompt", type=str, default="Vilket parti har minst antal deltagare i debatten, och vilket har flest?")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Choose model (gpt-4o, gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--temperature", type=float, default=0, help="Set temperature (0-1)")
    parser.add_argument("--system_message", type=str, default="""Translate the user's question so search in a  Neo4j
                         database debates from the swedish parliament (riksdagen). Read the question carefully and 
                        make sure you don't miss any details.""", help="Set the system message")

    args = parser.parse_args()

    models = ["gpt-4o", "gpt-4o-mini"]
    temperatures = [0, 0.3, 0.7]
    system_messages = [
        """
        Translate the user's question so search in a  Neo4j database debates from the swedish parliament (riksdagen). 
        Read the question carefully and make sure you don't miss any details.
        
        Node properties:
        Person {party: STRING, gender: STRING, name: STRING}
        Statement {text: STRING}
        Party {partyName: STRING}
        Relationship properties:

        The relationships:
        (:Person)-[:STATED]->(:Statement)
        (:Person)-[:BELONGS_TO]->(:Party)

        Here is an example of a good cypher query:
        MATCH (p:Person)-[:BELONGS_TO]->(party:Party {partyName: 'Centerpartiet'})
        RETURN p.name, p.gender
        """,
        """
        Your job is to respond to natural language queries in Swedish. Translate these
        queries to Cypher queries to respond to questions based on data from a particular debate 
        in the Swedish Parliament.
        """
    ]
    tester = CypherModelTester()

    model_results = tester.test_different_models(args.prompt, models, args.temperature, args.system_message)
    temp_results = tester.test_different_temperatures(args.prompt, args.model, temperatures, args.system_message)
    sys_msg_results = tester.test_different_system_messages(args.prompt, args.model, args.temperature, system_messages)

    tester.print_results(model_results, temp_results, sys_msg_results)
    #Stester.save_results("test_results.json", model_results, temp_results, sys_msg_results)

