import os
import traceback
import logging
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex
from dotenv import load_dotenv
load_dotenv()
NEO4J_URL = os.getenv('NEO4J_URL')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

def load_graphindex():
    """Loads an existing PropertyGraphIndex from Neo4j."""  # inserted
    print('\nConnecting to Neo4j and loading existing Knowledge Graph Index...')
    pg_store = Neo4jPropertyGraphStore(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)
    embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    llm = GoogleGenAI(model='gemini-2.5-pro')
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 10
    storage_context = StorageContext.from_defaults(graph_store=pg_store)
    pg_index = PropertyGraphIndex.from_existing(property_graph_store=pg_store)
    print('\nOK PropertyGraphIndex loaded successfully from Neo4j.')
    return (pg_index, pg_store)

def main():
    print('======================================================================')
    print('NASA BIOSCIENCE - LOAD EXISTING GRAPH (Neo4j + LlamaIndex)')
    print('======================================================================')
    try:
        index, graph_store = load_graphindex()
        driver = graph_store._driver
        with driver.session() as session:
            pass  # postinserted
    except Exception as e:
            node_count = session.run('MATCH (n) RETURN count(n) AS c').single()['c']
            edge_count = session.run('MATCH ()-[r]->() RETURN count(r) AS c').single()['c']
            print(f'\nTotal nodes in Neo4j: {node_count}')
            print(f'Total relationships in Neo4j: {edge_count}')
            print(f'\nERROR: {e}')
            traceback.print_exc()
if __name__ == '__main__':
    main()
