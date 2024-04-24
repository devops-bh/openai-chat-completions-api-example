"""
tried to reframe from writing an endpoint, 
first tried using processes to execute a python script which spat back out the embeddings as its output 
to be consumed by the initial NodeJS process
Made some progress but code was starting to pretty bad and I ran into an elusive behavior I wasn't expecting 

I then tried using the npm package equivalent of the python package I originally used for the embeddings 
https://www.npmjs.com/package/@tensorflow-models/universal-sentence-encoder
But it wasn't as straightforward as the Python script, and ended up having to try peer into the Github source code 
as the documentation is likely dated 
Also it seems like they were using a Python based encoder model, and thus were converting it into a TFJS model 
Which is fairly easy but I'd assume it'd be daunting/confusing if you didn't have prior context as to why 
developers do this 

Technically we could have a shared file, not sure if thats possible or if we'd need to keep re-opening 
and closing the file due to file locking mechanisms (which prevent race conditions) 
Though given that Numpy is a giant array perhaps a CSV could make sense? 

Ultimately you could imagine this script as a microservice or a serverless function, I am just going to 
refer to it as an endpoint 
"""

# https://www.youtube.com/watch?v=iWS9ogMPOI0
import tensorflow_hub as hub
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from numpy.linalg import norm

# remember to pip install uvicorn && uvicorn cosine-similarity-endpoint:app --port 8001 --reload (--reload means saving auto reruns the app)
print("curl http://127.0.0.1:8001/?cosine-similarity?current_question=...&past_question=...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# this is the cache 
past_questions_and_responses = []


def cosine_similarity(embedding_a, embedding_b):
    return np.dot(embedding_a, embedding_b) / (norm(embedding_a) * norm(embedding_b))

@app.get("/")
def root():
    return {"status": "running"}

# sadly this is quite inefficient as this will be done for each question 
# but there's likely ways to improve this e.g. having a cluster of nodes, 
# batching etc 
# offcourse it'd maybe be better to just use Python enitrely & use ChatGPT Python library 
# or have NodeJS share the ChatGPT response 
# but this will suffice for demonstrating the concept 
# offcourse as it is now could be stored/cached where the NodeJS API first tries the cache 
"""
I am not sure why, but 
but when curling the API, I kept getting an error, despite code correctness, 
even on Git Bash (Windows), so there may be a weird bug related to Window's CURL implementation and 
perhaps FastAPI, I should try the equivalent code using Flask or so, but I don't recall encountering 
this issue before 
Ultimately it works using fetch, but when curling the same URL as fetch, it does not work 
"""
@app.get("/similarity/")
def similarity(current: str, past: str):
    print("Current: ", current)
    print("Past: ", past)
    embeddings = embed([current, past])
    print( str(cosine_similarity(embeddings[0], embeddings[1])))
    return { "similarity": str(cosine_similarity(embeddings[0], embeddings[1]))}