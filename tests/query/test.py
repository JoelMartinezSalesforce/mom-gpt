
from icecream import ic
from pymilvus import MilvusClient, Collection, connections

from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.query.main.query_main import Query

if __name__ == '__main__':
    collection_name = 'health_data_cons_final'

    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    collection = Collection(collection_name)

    encoder = JSONEncoder \
        ("/Users/isaacpadilla/milvus-dir/mom-gpt/services/model/embeddings/bare_testing/dump/network_health_cons.json")

    query = Query(collection_name, encoder)

    prompt = "Can you give me data for POD106 for 2024 05 08"

    res = query.perform_query(prompt)

    ic(f"Searching with prompt: {prompt}")

    ic(res)

    ic(type(res))
