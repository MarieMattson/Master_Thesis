"""
input: natural language query
output: node
"""

import argparse

from llm_retrieval.cypher_translation import translate
from llm_retrieval.node_retrieval import retrieve_node

def wrapper(query:str)->list[str]:
    try:
        print("hello")
        cypher_query = translate(query)
        print(cypher_query)
        names = retrieve_node(cypher_query)
        print(names)
    except Exception as e:
        print(f"Something went wrong: {e}")

def main():
    parser = argparse.ArgumentParser(description="Execute a Cypher query and retrieve node data.")
    parser.add_argument("query", type=str, help="The Cypher query to execute")    
    args = parser.parse_args()
    
    wrapper(args.query)

if __name__ == "__main__":
    main()
