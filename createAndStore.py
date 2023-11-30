#put the file in the code in line no.33
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

# Load pre-trained model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# create embedding 
def get_embedding_batch(texts):
    texts = texts.tolist()
    encoded_inputs = tokenizer.batch_encode_plus(texts, padding='longest', truncation=True, max_length=512, return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)[0]
    return last_hidden_states.mean(1).numpy()

def process_column(column):
    column = column.astype(str)
    with Pool() as pool:
        embeddings = list(tqdm(pool.imap(get_embedding_batch, np.array_split(column, pool._processes)), total=pool._processes))
    embeddings = [embedding.tolist() for batch in embeddings for embedding in batch]
    return embeddings

# Load the data
df = pd.read_csv("Book1.csv", on_bad_lines='skip') #paste the file path
df = df.dropna(subset=['product'])

# Process each column in parallel
for column in tqdm(['product', 'category', 'sub_category', 'brand', 'type', 'description']):
    df[column + '_embedding'] = process_column(df[column])

# Combine the embeddings
df['combined_embedding'] = df.apply(lambda row: np.concatenate([row['product_embedding'], row['category_embedding'], row['sub_category_embedding'], row['brand_embedding'], row['type_embedding'], row['description_embedding']]), axis=1)

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://ce08a85c-c7d5-482f-b897-0ebbd35940b0.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="J9D_YF2h6s33XRY-20xaHbQurQCEIvV35uGqhAOIqCnl79TDGnbQNg",
)
# only uncomment if you want to create new collection and hence 
# make the changes at other lines as well.
# qdrant_client.create_collection(
#     collection_name="bigbasket1",
#     vectors_config=VectorParams(size=4608, distance=Distance.DOT),
# )

# Upsert in batches
batch_size = 32
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    points = [{
        "id": index,
        "vector": row['combined_embedding'].tolist(),
        "payload": {"product": row['product']}
    } for index, row in batch.iterrows()]
    qdrant_client.upsert(collection_name="bigbasket1", points=points)