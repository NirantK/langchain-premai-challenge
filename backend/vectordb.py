import os
import uuid
from typing import List

import pandas as pd
from fastapi.responses import JSONResponse
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Qdrant
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (CollectionStatus, Distance, PointStruct,
                                       UpdateStatus, VectorParams)
from transformers import AutoTokenizer

# IP_ADDRESS = "http://3.91.215.30"
IP_ADDRESS = "http://localhost"
# OPENAI_API_BASE = "http://3.91.215.30:8111/v1"
OPENAI_API_BASE = "http://localhost:8111/v1"
# EMBEDDING_ADDRESS = "http://3.91.215.30:8444/v1"
EMBEDDING_ADDRESS = "http://localhost:8444/v1"
class VectorDatabase:
    """VectorDatabase class initializes the Vector Database index_name and loads the dataset
    for the usage of the subclasses."""

    def __init__(self, collection_name: str, top_k: int = 3):
        self.collection_name = collection_name
        logger.info(f"Index name: {self.collection_name} initialized")
        # Load the dataset
        self.dataframe = pd.read_csv(
            "data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
        )
        logger.info(f"Dataset loaded with {self.dataframe.shape[0]} records")
        self.top_k = top_k
        self.vector_dimension = 384

        # embedding config - using All MiniLM L6 v2
        os.environ["OPENAI_API_KEY"] = "random-string"
        self.embeddings = OpenAIEmbeddings(openai_api_base=f"{EMBEDDING_ADDRESS}")
        logger.info("OpenAI Embeddings initialized")

    def upsert(self) -> str:
        raise NotImplementedError

    def query(self, query_embedding: List[float]) -> dict:
        raise NotImplementedError


class QdrantDB(VectorDatabase):
    """QdrantDB class is a subclass of VectorDatabase that
    interacts with the Qdrant Cloud Vector Database. It has the following methods:
    - upsert: Upserts the dataset into the Qdrant collection(index) with the payload(metadata)
    - query: Queries the Qdrant collection(index) with the query embedding along with
    the payload(metadata)
    - delete_index: Deletes the Qdrant collection(index)
    """

    def __init__(self, collection_name: str):
        super().__init__(collection_name)
        self.client = QdrantClient(IP_ADDRESS, port=6333)
        logger.info(f"QdrantDB initialized with index name: {self.collection_name}")

        qdrant_collections = self.client.get_collections()

        # If no collections exist or if the index_name is not present in the collections, create the collection
        if len(qdrant_collections.collections) == 0 or not any(
            self.collection_name in collection.name
            for collection in qdrant_collections.collections
        ):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension, distance=Distance.COSINE
                ),
            )
            logger.info(f"Qdrant collection {self.collection_name} created")

    def upsert(self) -> str:
        logger.info(f"total vectors from upsert: {self.dataframe.shape[0]}")
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        logger.info("Computing the tokenized length of each instruction")
        # compute the tokenized length of each instruction
        self.dataframe["Instructions_tokenized_length"] = self.dataframe[
            "Instructions"
        ].apply(lambda x: len(tokenizer.tokenize(str(x))))

        logger.info("Filtering the dataset")

        # drop all NaN values
        self.dataframe = self.dataframe.dropna()

        # drop the rows where the Instructions_tokenized_length is greater than 2000 and greater than 1
        self.dataframe = self.dataframe[
            (self.dataframe["Instructions_tokenized_length"] < 2000)
            & (self.dataframe["Instructions_tokenized_length"] > 1)
        ]
        logger.info(f"Total vectors after filtering: {self.dataframe.shape[0]}")

        # vector and payloads as points for qdrant
        docs=[]

        for index, row in self.dataframe.iterrows():
            docs.append(Document(
                        page_content=row["Title"], metadata={"recipe": row["Instructions"], "image": f"{row['Image_Name']}.jpg"}
                    ))
            
        # upsert the vectors and payloads into the qdrant collection
        self.vectorstore = Qdrant.from_documents(
            docs,
            self.embeddings,
            url=f"{IP_ADDRESS}:6333",  # Qdrant gRPC API endpoint
            collection_name=self.collection_name,
        )

        return "Upserted successfully"

    def query(self, query: str) -> dict:
        # Qdrant Output:
        # {
        #     "result": [
        #         {"id": 4, "score": 1.362},
        #         {"id": 1, "score": 1.273},
        #         {"id": 3, "score": 1.208},
        #     ],
        #     "status": "ok",
        #     "time": 0.000055785,
        # }
        self.vectorstore = Qdrant(client=self.client, collection_name=self.collection_name, embeddings=self.embeddings)

        # chat completion llm
        llm = ChatOpenAI(
            openai_api_base=f"{OPENAI_API_BASE}", temperature=0.2, max_tokens=128
        )

        # Using OpenAI directly
        # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=qdrant_docsearch.as_retriever())

        # Using Vicuna via Premai app 
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.vectorstore.as_retriever(), return_source_documents=True)

        result = qa({"query": query})

        print(result)
        return "hello"
        # return JSONResponse(content=result)

    def delete_index(self) -> str:
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Qdrant collection {self.collection_name} deleted")
        logger.info(self.client.get_collections())
        return "Index deleted"
