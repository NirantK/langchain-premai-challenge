from typing import Type

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from vectordb import QdrantDB, VectorDatabase

# Initializations
app = FastAPI()

# Define the collection name
collection_name = "recipe_title_collection"
vector_db = None


# Define the request model
class QueryRequest(BaseModel):
    query: str


# Dependency function to choose a vector database implementation
def get_vector_db() -> Type[VectorDatabase]:
    # Choose either PineconeDatabase or QdrantDatabase here
    vector_db_class = QdrantDB
    return vector_db_class(collection_name)


@app.on_event("startup")
async def startup_event():
    vector_db = get_vector_db()
    # vector_db.upsert()


@app.post("/ask")
async def ask(
    request: QueryRequest, vector_db: VectorDatabase = Depends(get_vector_db)
) -> dict:


    if request.query:
        # Query the VectorDatabase
        result = vector_db.query(request.query)
    else:
        pass

    return {"result": result}


@app.post("/upsert")
async def upsert():
    vector_db = get_vector_db()
    return vector_db.upsert()


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/delete")
async def delete():
    vector_db = get_vector_db()
    return vector_db.delete_index()


# @app.on_event("shutdown")
# async def shutdown():
#     vector_db = get_vector_db()
# #     vector_db.delete_index()