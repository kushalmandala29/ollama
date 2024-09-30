from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import  ChatOllama, OllamaEmbeddings
import json
import os
import logger
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
# from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import JSONLoader
from pathlib import Path

username = 'elastic'
password = 'kushal@123'
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=(username, password)
)






# response = requests.post('http://localhost:11434/api/embeddings', json={
#         "model": 'tinyllama',
#         "prompt": "Given a chat history and the latest user question "
#     })
# # print(response.json)

# embedding_data = response.json()
# if 'embedding' in embedding_data:
#     print(embedding_data['embedding'])
# embeddings= OllamaEmbeddings(
#     model="tinyllama"
# )

# Connect to Elasticsearch
# es = Elasticsearch(['http://localhost:9200'])

# Function to generate embeddings
# Function to generate embeddings using Ollama
def generate_embedding(text, model="tinyllama"):
    try:
        response = requests.post('http://localhost:11434/api/embeddings', json={
            "model": model,
            "prompt": text
        })
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        embedding_data = response.json()
        if 'embedding' in embedding_data:
            return embedding_data['embedding']
        else:
            logger.error(f"Unexpected response format: {embedding_data}")
            raise ValueError("Embedding not found in response")
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse API response: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_embedding: {e}")
        raise

# Define the index name
index_name = 'college_info'

# Define Elasticsearch mapping
mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "meta_description": {"type": "text"},
            "keywords": {"type": "text"},
            "about": {"type": "text"},
            "courses": {
                "type": "nested",
                "properties": {
                    "Course": {"type": "text"},
                    "Details": {"type": "text"}
                }
            },
            "fees_eligibility": {
                "type": "nested",
                "properties": {
                    "Course": {"type": "text"},
                    "Fees": {"type": "text"},
                    "Eligibility": {"type": "text"}
                }
            },
            "cutoff": {
                "type": "nested",
                "properties": {
                    "Courses": {"type": "text"},
                    "Round 1": {"type": "keyword"},
                    "Round 2": {"type": "keyword"}
                }
            },
            "faqs": {
                "type": "nested",
                "properties": {
                    "question": {"type": "text"},
                    "answer": {"type": "text"}
                }
            },
            "title_vector": {"type": "dense_vector", "dims": 384},
            "description_vector": {"type": "dense_vector", "dims": 384},
            "about_vector": {"type": "dense_vector", "dims": 384},
            "combined_vector": {"type": "dense_vector", "dims": 384}
        }
    }
}
json_directory = 'main/dataset/A. M. Reddy Memorial College of Engineering and Technology.json'  # Update this to your JSON files directory


# Create or update the index with mapping
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)
else:
    es.indices.put_mapping(index=index_name, body=mapping["mappings"])

# Function to process a single JSON file using LangChain's JSONLoader
def process_json_file(json_directory):
    with open(json_directory, 'r') as file:
        data = json.load(file)
        print(type(data))

    combined_text = f"{data['title']} {data['meta_description']} {data['Section']['description']}"
    combined_vector = generate_embedding(combined_text)
    combined_vector

    # Prepare document for Elasticsearch
    doc = {
        'title': data['title'],
        'meta_description': data['meta_description'],
        'keywords': data['keywords'],
        'about': data['Section']['description'],
        'courses': data['A.M.Reddy Memorial College of Engineering and Technology, Andhra Pradesh: Courses Offered'],
        'fees_eligibility': data['AM Reddy Memorial College of Engineering and Technology Fees & Eligibility'],
        'cutoff': data['AM Reddy Memorial College of Engineering and Technology, AP EAPCET Cutoff 2024'],
        'faqs': data['faqs'],
        'title_vector': generate_embedding(data['title']),
        'description_vector': generate_embedding(data['meta_description']),
        'about_vector': generate_embedding(data['Section']['description']),
        'combined_vector': combined_vector
    }
    print(doc)
    return doc
# Function to generate actions for bulk indexing
def generate_actions(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            doc = process_json_file(file_path)
            yield {
                "_index": index_name,
                "_source": doc
            }

# Directory containing JSON files
json_directory = 'main/dataset/A. M. Reddy Memorial College of Engineering and Technology.json'  # Update this to your JSON files directory
yy=process_json_file(json_directory)
# Perform bulk indexing
success, failed = bulk(es, generate_actions(json_directory))

print(f"Successfully indexed {success} documents.")
print(f"Failed to index {len(failed)} documents.")

# Refresh the index
es.indices.refresh(index=index_name)

print("Indexing complete and index refreshed.")