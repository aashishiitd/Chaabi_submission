# Chaabi_submission
A contextual query Engine for Bigbasket products dataset. 

This respo. has two files as per the assignment.

- First file just contains the code to tokenize and create embedding(using BERT pretrained model) and upload the embedding with their paylaod as product name.
- Second file has the other two parts as assigned in the assignment.In this,first the api passes the query to the function in which the query string tokenizes and embedding is created and after it does the semantic search with the datapoints in Qdrant DB and finds the top five product according to the query and passes these to GPT2 pretrained model to answer the contextual query. 

Setup-

