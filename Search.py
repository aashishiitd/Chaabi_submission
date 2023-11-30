from transformers import GPT2LMHeadModel, GPT2Tokenizer
from qdrant_client.http.models import SearchRequest, Filter, Distance
from qdrant_client.http.models import VectorParams
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
import torch
import numpy as np

# Load pre-trained model for embeddings
tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
model_bert = AutoModel.from_pretrained('bert-base-uncased')

def get_embedding_batch(texts):
    # Encode text
    input_ids = torch.tensor([tokenizer_bert.encode(texts, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model_bert(input_ids)[0]
    # Take the mean of the hidden states to get the context of the sentence
    return last_hidden_states.mean(1).numpy().flatten()

# Load pre-trained model for response generation
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(input_query):
    # Encode the input query
    input_ids = tokenizer_gpt2.encode(input_query, return_tensors='pt')

    # Generate a response
    output = model_gpt2.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, do_sample=True)

    # Decode the response
    response = tokenizer_gpt2.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://ce08a85c-c7d5-482f-b897-0ebbd35940b0.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="J9D_YF2h6s33XRY-20xaHbQurQCEIvV35uGqhAOIqCnl79TDGnbQNg",
)

def get_product(query):
    # Get the embedding for the query
    query_embedding = get_embedding_batch(query)
    query_embedding = np.concatenate([query_embedding] * 6).tolist()

    # Search for the most similar vectors in the database
    search_result = qdrant_client.search(
        collection_name="bigbasket1",
        query_vector=query_embedding,
        limit=5,
    )
    response=[]
    # response.append("-------------------------\n The top five matches are - \n")
    for i, point in enumerate(search_result):
        response.append(f"{point.payload['product']}\n")
    return response

def get_contextual_answers(query):
    # Get the embedding for the query
    query_embedding = get_embedding_batch(query)
    query_embedding = np.concatenate([query_embedding] * 6).tolist()

    # Search for the most similar vectors in the database
    search_result = qdrant_client.search(
        collection_name="bigbasket1",
        query_vector=query_embedding,
        limit=5,
    )

    # Generate a response for each similar vector
    responses = []
    # responses.append(f"-------------------\n The context query responses are- \n")
    for i, point in enumerate(search_result):
        # print(point.payload['product']+'\n')
        response = generate_response(point.payload['product'])
        responses.append(f"\nResponse {i + 1}:\n{response}\n")

    return responses

# responses = get_contextual_answers("What is the best soap?")
# response = get_product("What is the best soap?")
# for item in response:
#     print(item)
# for response in responses:
#     print(response)
