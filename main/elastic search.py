import json
import torch
import os
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/embeddings"

# Connect to Elasticsearch
username = 'elastic'
password = 'kushal@123'
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=(username, password)
)
es.info()

print(es.ping())

# Function to generate embeddings using Ollama
def generate_embedding(text, model="tinyllama"):
    try:
        response = requests.post(OLLAMA_API, json={
            "model": model,
            "prompt": text
        })
        response.raise_for_status()
        # print(response.json())
        embedding_data = response.json()
        if 'embedding' in embedding_data:
            return embedding_data['embedding']
        else:
            logger.error(f"Unexpected response format: {embedding_data}")
            raise ValueError("Embedding not found in response")
    except Exception as e:
        logger.error(f"Error in generate_embedding: {e}")
        raise


print(generate_embedding("Hello, how are you?"))

# Define the index name
main_name = 'college_info'
embedding_index = 'college_embeddings'

# Define a more flexible Elasticsearch mapping
main_mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "meta_description": {"type": "text"},
            "keywords": {"type": "text"},
            "about": {"type": "text"},
            "courses": {
                "type": "nested",
                "properties": {
                    "name": {"type": "text"},
                    "details": {"type": "text"}
                }
            },
            "fees_eligibility": {
                "type": "nested",
                "properties": {
                    "course": {"type": "text"},
                    "fees": {"type": "text"},
                    "eligibility": {"type": "text"}
                }
            },
            "cutoff": {
                "type": "nested",
                "properties": {
                    "course": {"type": "text"},
                    "cutoff_value": {"type": "float"}
                }
            },
            "faqs": {
                "type": "nested",
                "properties": {
                    "question": {"type": "text"},
                    "answer": {"type": "text"}
                }
            },
            # "combined_vector": {"type": "dense_vector", "dims": 4096}
        }
    }
}

embedding_mapping = {
    "mappings": {
        "properties": {
            "college_id": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims":4096 }
        }
    }
}

# Create or update the index with mapping
for index, mapping in [(main_name, main_mapping), (embedding_index, embedding_mapping)]:
    if not es.indices.exists(index=main_name):
        es.indices.create(index=main_name, body=mapping)
    else:
        es.indices.put_mapping(index=main_name, body=mapping["mappings"])

def extract_courses(data):
    courses_key = next((key for key in data.keys() if 'Courses' in key), None)
    if courses_key:
        courses = data[courses_key]
        return [{'name': course['Course'], 'details': course.get('Details', '')} for course in courses]
    return []

def extract_cutoff(data):
    cutoff_key = next((key for key in data.keys() if 'Cutoff' in key), None)
    if cutoff_key:
        cutoff_data = data[cutoff_key]
        return [{'course': item['Courses'], 'cutoff_value': float(item['Round 1'])} for item in cutoff_data]
    return []

def extract_fees_eligibility(data):
    fees_key = next((key for key in data.keys() if 'Fees' in key), None)
    if fees_key:
        fees_data = data[fees_key]
        return [{'course': item['Course'], 'fees': item['Fees'], 'eligibility': item['Eligibility']} for item in fees_data]
    return []

# Function to process a single JSON file
def process_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    try:
        # Extract relevant information
        title = data.get('title', '')
        meta_description = data.get('meta_description', '')
        keywords = data.get('keywords', '')
        about = data['Section'].get('description', '') if 'Section' in data else ''

        courses = extract_courses(data)
        cutoff = extract_cutoff(data)
        fees_eligibility = extract_fees_eligibility(data)

        # Generate combined text for embedding
        combined_text = f"{title} {meta_description} {about}"
        combined_vector = generate_embedding(combined_text)

        # Prepare document for Elasticsearch
        doc = {
            'title': title,
            'meta_description': meta_description,
            'keywords': keywords,
            'about': about,
            'courses': courses,
            'cutoff': cutoff,
            'fees_eligibility': fees_eligibility,
            'faqs': data.get('faqs', []),
            # 'combined_vector': combined_vector
        }

        
        return doc,combined_vector
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise

# Function to generate actions for bulk indexing
def generate_actions(directory):
    print(len(os.listdir(directory)))
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                main_doc, embedding = process_json_file(file_path)
                
                # Yield action for main index
                yield {
                    "_index": main_name,
                    "_source": main_doc
                }

                # Yield action for embedding index
                yield {
                    "_index": embedding_index,
                    "_source": {
                        "college_id": filename,  # Use filename or another unique identifier
                        "embedding": embedding
                    }
                }
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    

# Directory containing JSON files
json_directory = 'main/dataset'  # Update this to your JSON files directory

# Perform bulk indexing
success, failed = bulk(es, generate_actions(json_directory))

print(f"Successfully indexed {success} documents.")
print(f"Failed to index {len(failed)} documents.")

# Refresh the index
es.indices.refresh(index=main_name)

print("Indexing complete and index refreshed.")

# Function to perform combined text and semantic search
def combined_search(query, fields=['title', 'meta_description', 'about']):
    query_vector = generate_embedding(query)
    
#     search_query = {
#     "query": {
#         "bool": {
#             "should": [
#                 {
#                     "multi_match": {
#                         "query": query,
#                         "fields": fields
#                     }
#                 },
#                 {
#                     "script_score": {
#                         "query": {"match_all": {}},
#                         "script": {
#                             "source": "cosineSimilarity(params.query_vector, doc['combined_vector']) + 1.0",
#                             "params": {"query_vector": query_vector}
#                         }
#                     }
#                 }
#             ]
#         }
#     }
# }
    search_query = {
    "query": {"match_all": {}},
    "_source": ["embedding"],  # Adjust this field name if your embedding field is named differently
    "size": 2  # Limiting to 1 document for this example
}
    results = es.search(index=main_name, body=search_query)
    return results['hits']['hits']

# Example usage of combined search
search_queries = [
    'AAA COLLEGE OF ENGINEERING AND TECHNOLOGY',
    # "Computer Science courses in top colleges",
    # "Affordable engineering programs"
]
# print(combined_search(search_queries[0]))
response = combined_search(search_queries[0])
if response:
    print(response[0])

for query in search_queries:
    logger.info(f"\nCombined Search Results for: '{query}'")
    try:
        search_results = combined_search(query)
        for hit in search_results:
            print(hit)
            logger.info(f"Score: {hit['_score']}, Title: {hit['_source']['title']}")
            logger.info(f"Courses: {[course['name'] for course in hit['_source']['courses']]}")
            logger.info("---")
    except Exception as e:
        logger.error(f"Error during search for query '{query}': {e}")








mapping = es.indices.get_mapping(index=main_name)
print("Index Mapping:")
print(mapping)