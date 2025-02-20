import os
import argparse
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

# Initialize Neo4j graph
graph = Neo4jGraph()
graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)

def test_model(prompt, model="gpt-4o", temperature=0, system_message=""):
    """Test different prompts, models, temperatures, and system messages."""
    llm = ChatOpenAI(model=model, temperature=temperature, openai_api_key=OPEN_API_KEY)
    chain = GraphCypherQAChain.from_llm(
        graph=enhanced_graph, llm=llm, verbose=True, allow_dangerous_requests=True,
        return_intermediate_steps=True
    )

    messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]

    response = chain.invoke({"query": prompt, "messages": messages})
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test different prompts, models, and parameters.")
    parser.add_argument("--prompt", type=str, default="Vilket parti har minst antal deltagare i debatten, och vilket har flest?")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Choose model (gpt-4o, gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--temperature", type=float, default=0, help="Set temperature (0-1)")
    parser.add_argument("--system_message", type=str, default="Your job is to respond to natural language queries in Swedish. Translate these queries to Cypher queries to respond to questions based on data from a particular debate in the Swedish Parliment.", help="Set the system message")

    args = parser.parse_args()

    print(f"Testing with model: {args.model}, temperature: {args.temperature}, system message: {args.system_message}")
    response = test_model(args.prompt, args.model, args.temperature, args.system_message)

    print(graph.query("MATCH (p:Person) WHERE p.name = 'Gunnar Sträng' RETURN p AS Person"))
    print(response)
