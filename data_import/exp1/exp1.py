from llm_retrieval.full_pipeline_class import GraphRAG
from datasets import load_dataset

graph_rag = GraphRAG()
test_set =  load_dataset("json", data_files="/mnt/c/Users/User/thesis/data_import/exp1/first_test_set.json", split="train")

def run_test(user_query):

    print("Translating user query into Cypher query...")
    cypher_query = graph_rag.translate_to_cypher(user_query)
    print("Generated Cypher Query:", cypher_query)

    print("\nRetrieving nodes from Neo4j...")
    retrieved_nodes = graph_rag.retrieve_nodes(cypher_query)
    print(f"Retrieved {len(retrieved_nodes)} nodes.")

    print("\nRanking nodes by similarity...")
    ranked_nodes = graph_rag.rank_nodes_by_similarity(user_query, retrieved_nodes)

    print("\nPrinting ranked nodes...")
    graph_rag.print_ranked_nodes(ranked_nodes)

    print("\nGenerating final response...")
    final_response = graph_rag.generate_response(ranked_nodes, user_query)
    print("\nFinal Response:\n", final_response)


if __name__=="__main__":

    extra_prompt = """
                    Du måste returnera a.anforande_text, c.text, c.chunk_id och c.embedding! 
                    Generera en Cypher query som begränsar till den aktuella talarens anförande, 
                    men undvik för många filter."""
    for qa_pair in test_set:
        run_test(str(extra_prompt + qa_pair["question"]))
