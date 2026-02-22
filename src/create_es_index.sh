#!/usr/bin/env bash
set -e




# create_es_index.sh
#
# Create the Elasticsearch index used for PubMed document ingestion.
#
# IMPORTANT:
# This script assumes Elasticsearch is running locally via Docker,
# and is accessible at:
#
#   http://localhost:9200
#
# Typical way to start Elasticsearch with Docker:
#
#   docker run -d \
#     --name elasticsearch \
#     -p 9200:9200 \
#     -e "discovery.type=single-node" \
#     docker.elastic.co/elasticsearch/elasticsearch:8.11.1
#
# If you run Elasticsearch elsewhere (remote server, different port,
# authentication enabled, etc.), update ES_HOST accordingly.
#




ES_HOST="http://localhost:9200"
INDEX_NAME="pubdocs"


echo "[INFO] Creating Elasticsearch index: ${INDEX_NAME}"

# field mappings and settings
curl -X PUT "${ES_HOST}/${INDEX_NAME}" \
  -H "Content-Type: application/json" -d'
{
  "settings": { "number_of_shards": 1, "number_of_replicas": 0 },
  "mappings": {
    "properties": {
      "doc_id": { "type": "keyword" },
      "interventions": { "type": "text", "analyzer": "english" },
      "primary_endpoint": { "type": "text", "analyzer": "english" },
      "pub_text": { "type": "text", "analyzer": "english" }
    }
  }
}'



echo
echo "[INFO] Index '${INDEX_NAME}' created successfully."