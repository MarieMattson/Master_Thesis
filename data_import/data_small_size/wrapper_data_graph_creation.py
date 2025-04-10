import json
from data_import import filter_dataset_by_year
from data_import import normalise_data
from data_import import create_graph
from data_import import embed_nodes

full_dataset = "/mnt/c/Users/User/thesis/data_import/full_dataset.csv"
output_file = "/mnt/c/Users/User/thesis/data_import/data_small_size/dataset_small.json"
years_for_small_dataset = ["2020/21", "2021/22"]

#filter_dataset_by_year.filter_and_save_by_year(years=years_for_small_dataset, input_file=full_dataset, output_file=output_file)

#with open(output_file, 'r', encoding='utf-8') as f:
#        filtered_data = json.load(f)

#for item in filtered_data:
#    if 'talare' in item:
#        item['talare'] = normalise_data.Speaker(**item).talare

#with open(output_file, 'w', encoding='utf-8') as f:
#        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

#GraphCreator = create_graph.GraphCreator()
#data_for_graph = GraphCreator.load_json(output_file)
#GraphCreator.test_connection()  # Should print: Hello, Neo4j!
#GraphCreator.insert_data_into_neo4j(data_for_graph)

NodeEmbedder = embed_nodes.ChunkEmbedder()
NodeEmbedder.create_embedded_chunks()