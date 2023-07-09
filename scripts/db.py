from typing import List

import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, 
                                       VectorParams,
                                       PointStruct, 
                                       UpdateStatus, 
                                       Filter, 
                                       FieldCondition, 
                                       MatchValue)

class VectorDB():
    def __init__(self, collection_name: str):
        self.client = QdrantClient("http://vectordb:6333")
        self.col_name = collection_name
        
    def get_current_idx(self):
        return self.client.get_collection(collection_name=self.col_name).points_count
        
    def create_collection(self, embedding_size: int):
        try:
            collection_info = self.client.get_collection(collection_name=self.col_name, timeout=1)
            print(f"Collection <{self.col_name}> Already exists")
        except qdrant_client.http.exceptions.UnexpectedResponse:
        #    # create client
            print("Create collection: ", self.col_name)
            self.client.recreate_collection(
                                        collection_name=self.col_name,
                                        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
                                        timeout=1
                                        )
        
        # check client exists
        collection_info = self.client.get_collection(collection_name=self.col_name)
        print("Col info: \n", collection_info)
        
    def add_face_emb(self, face_embedding: List[float], user_name: str):
        operation_info = self.client.upsert(
                                collection_name=self.col_name,
                                wait=True,
                                points=[
                                    PointStruct(id=self.get_current_idx() + 1, vector=face_embedding, payload={"user": user_name}),
                                ]
                            )
        
        print(operation_info)
        # check if the operation succeeded
        assert operation_info.status == UpdateStatus.COMPLETED
        
        return operation_info

    def verify_user(self, face_embedding: List[float], user_name: str, threshold: float = 0.5):
        search_result = self.client.search(
                collection_name=self.col_name,
                query_vector=face_embedding, 
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user",
                            match=MatchValue(value=user_name)
                        )
                    ]
                ),
                limit=3
            )
        
        print(search_result)
        
        for result in search_result:
            if(result.score > threshold):
                return True
        return False

if __name__ == '__main__':
    import numpy as np
    
    db = VectorDB("test_v2")
    db.create_collection(embedding_size=100)
    
    rd_vector = np.random.rand(100)
    query_vector = np.random.rand(100)
    
    db.add_face_emb(rd_vector.tolist(), "son")
    db.add_face_emb(query_vector.tolist(), "son")
    
    # db.verify_user(rd_vector.tolist(), "son")
    # db.verify_user(query_vector.tolist(), "nam")

    
                                    
