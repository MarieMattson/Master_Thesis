"""
input: natural language query
output: node
"""

import argparse 

from llm_retrieval.old.generate_response import response_generation
from llm_retrieval.old.cypher_translation import translate
from llm_retrieval.old.node_ranking import rank_nodes_by_similarity
from llm_retrieval.old.node_retrieval import retrieve_node

def wrapper(query:str)-> tuple[str, list[str]]:
    try:
        cypher_query = translate(query)
        nodes = retrieve_node(cypher_query)
        return cypher_query, nodes    
    except Exception as e:
        print(f"Something went wrong: {e}")

#def main():
#    parser = argparse.ArgumentParser(description="Execute a Cypher query and retrieve node data.")
#    parser.add_argument("query", type=str, help="The Cypher query to execute")    
#    args = parser.parse_args()
#    
#    wrapper(args.query)

if __name__ == "__main__":
    user_query = (
        """
        Hur argumenterar JESSICA POLFJÄRD för sänkt restaurangmoms och fler jobb 
        i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text, c.chunk_id och c.embedding!
        Generera en Cypher query som begränsar till den specifika debatten 
        och den aktuella talarens anförande, men undvik för många filter.
        """) 
    cypher_query=translate(user_query)
    retrieved_nodes=retrieve_node(cypher_query)
    ranked_nodes=rank_nodes_by_similarity(query_text=user_query, retrieved_nodes=retrieved_nodes)
    final_response=response_generation(ranked_nodes=ranked_nodes, user_query=user_query)
    
    print("\n--- User Query ---")
    print(f"{user_query}\n")

    print("\n--- Translated Cypher Query ---")
    print(f"{cypher_query}\n")

    print("\n--- Generated Response ---")
    print(f"{final_response}\n")
