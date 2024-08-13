import unittest

from icecream import ic
from pymilvus import utility, connections, Collection

from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.query.main.query_main import Query


class TestMilvus(unittest.TestCase):
    collection_name = 'health_data_cons_final'

    def test_milvus_connection(self):
        is_connected = False
        try:
            connections.connect(
                alias="default",
                user='username',
                password='password',
                host='localhost',
                port='19530'
            )

            is_connected = True

            self.assertEqual(is_connected, True, "Server is not connected")
        except Exception as e:

            self.assertEqual(is_connected, False, f"Server is not connected with exception: {e}")

    def test_collection_information(self, collection_name: str = collection_name):
        # Connect to Milvus
        self.test_milvus_connection()

        exists = collection_name in utility.list_collections()

        self.assertEqual(exists, True, "Collection does not exist")

        collection = Collection(self.collection_name)

        self.assertEqual((collection.num_entities != 0 and collection.num_entities > 1000),
                         True, "Collection is Empty"), self.assertEqual(
            (len(collection.indexes) != 0), True, "Collection contains no indexes")

    def test_milvus_query1(self):

        self.test_milvus_connection()

        self.test_collection_information(self.collection_name)

        encoder = JSONEncoder(
            "Path to JSON file"
        )

        query = Query(self.collection_name, encoder)

        prompt = "Can yoy give me data for POD106 for 2024 05 07"

        res = query.perform_query(prompt)

        ic(f"Searching with prompt: {prompt}")

        ground_truth = {
            "period": "day",
            "instance": "2024-05-07",
            "site": "POD106",
            "metric": "internet-avail-cc",
            "end": "2024-05-08 00:00:00.000000+0000",
            "start": "2024-05-07 00:00:00.000000+0000",
            "updated": "2024-05-08 00:40:50.889000+0000",
            "percentage": 99.96335654085746,
            "percentage-EUROPE": 100.0,
            "percentage-NORTH_AMERICA": 99.94764397905759,
            "percentage-ASIA": 100.0,
            "power-p95": "NA",
            "power-max": "NA",
            "percentage-OCEANIA": "NA",
            "power-avg": "NA",
            "percentage-max": "NA",
            "sum": "NA",
            "percentage-p95": "NA",
            "max": "NA",
            "percentage-CHINA": "NA"
        }

        res_groung_truth_preprocessed = encoder.preprocess_single_text([ground_truth])

        # Testing if the best result (or ranked best result) is exactly what we expect to prompt engineer the LLM

        self.assertEqual(res[0]['entity']['data'], res_groung_truth_preprocessed, "Query failed data "
                                                                                  "does not match ground"
                                                                                  " truth")


if __name__ == '__main__':
    unittest.main()
