# Chaabi_submission
A contextual query Engine for Bigbasket products dataset. 

This respo. has four files as per the assignment.

- First file(createAndStore.py) just contains the code to tokenize and create embedding(using BERT pretrained model) and upload the embedding with their paylaod as product name.
- Second file(Search.py) has the second part as assigned in the assignment.In this, the query string tokenizes and embedding is created and after it does the semantic search with the datapoints in Qdrant DB and finds the top five product according to the query and passes these to GPT2 pretrained model to answer the contextual query. 
- Third file(api.py) has the api wrap part, this hosts the api and allows post request to the api.
- Fourt file(testapi.py) using this file you can test by writing your query in the query variable, this will call the post request to the API and return the output as the first top five matches in the search at the DB and contextual response from the GPT2 model (this needs more fine tuning using the dataset as it doesn't a very good response)
  
Setup-  (Please install libraries - Flask,qdrant_client, numpy, pandas, tqdm, transformers, multiprocessing,torch)
1) Download all the files in the same folder.
2) To use the createAndStore.py just copy the path of the bigbasket dataset's csv file and run using the command "python3 createAndStore.py" (in Linux env.) and "python createAndStore.py(in windows env.) 
3) To run the api and test the query Engine, first run the api.py file, now write the host address in the url variable of the testapi.py file and write your query on which you want to test and run testapi.py in other terminal.

Scope for Improvement:
  1) We can fine tune the GPT2 model to get more contextual answers based on the queries.  
      
