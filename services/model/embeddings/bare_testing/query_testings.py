from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import random
from pymilvus import AnnSearchRequest


if __name__ == '__main__':

    # Connect to Milvus
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    # Create schema
    fields = [
        FieldSchema(name="film_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="filmVector", dtype=DataType.FLOAT_VECTOR, dim=5),  # Vector field for film vectors
        FieldSchema(name="posterVector", dtype=DataType.FLOAT_VECTOR, dim=5)]  # Vector field for poster vectors

    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)

    # Create collection
    collection = Collection(name="test_collection", schema=schema)

    # Create index for each vector field
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }

    collection.create_index("filmVector", index_params)
    collection.create_index("posterVector", index_params)

    # Generate random entities to insert
    entities = []

    for _ in range(1000):
        # generate random values for each field in the schema
        film_id = random.randint(1, 1000)
        film_vector = [random.random() for _ in range(5)]
        poster_vector = [random.random() for _ in range(5)]

        # create a dictionary for each entity
        entity = {
            "film_id": film_id,
            "filmVector": film_vector,
            "posterVector": poster_vector
        }

        # add the entity to the list
        entities.append(entity)

    collection.insert(entities)

    # Create ANN search request 1 for filmVector
    query_filmVector = [
        [0.8896863042430693, 0.370613100114602, 0.23779315077113428, 0.38227915951132996, 0.5997064603128835]]

    search_param_1 = {
        "data": query_filmVector,  # Query vector
        "anns_field": "filmVector",  # Vector field name
        "param": {
            "metric_type": "L2",  # This parameter value must be identical to the one used in the collection schema
            "params": {"nprobe": 10}
        },
        "limit": 2  # Number of search results to return in this AnnSearchRequest
    }
    request_1 = AnnSearchRequest(**search_param_1)

    # Create ANN search request 2 for posterVector
    query_posterVector = [
        [0.02550758562349764, 0.006085637357292062, 0.5325251250159071, 0.7676432650114147, 0.5521074424751443]]

    search_param_2 = {
        "data": query_posterVector,  # Query vector
        "anns_field": "posterVector",  # Vector field name
        "param": {
            "metric_type": "L2",  # This parameter value must be identical to the one used in the collection schema
            "params": {"nprobe": 10}
        },
        "limit": 2  # Number of search results to return in this AnnSearchRequest
    }
    request_2 = AnnSearchRequest(**search_param_2)

    # Store these two requests as a list in `reqs`
    reqs = [request_1, request_2]

    from pymilvus import WeightedRanker

    # Use WeightedRanker to combine results with specified weights
    # Assign weights of 0.8 to text search and 0.2 to image search
    rerank = WeightedRanker(0.8, 0.2)

    collection.load()

    res = collection.hybrid_search(
        reqs,  # List of AnnSearchRequests created in step 1
        rerank,  # Reranking strategy specified in step 2
        limit=2  # Number of final search results to return
    )

    print(res)