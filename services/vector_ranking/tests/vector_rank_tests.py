import unittest

from icecream import ic

from services.vector_ranking.vector_ranking import VectorRanking


class TestVector(unittest.TestCase):
    vector_ranking = VectorRanking()

    def test_vector_ranking_health_data(self):
        prompt_to_rank_health_data: str = 'Hello Can you give me the data for site IA4 and explain the P95 for network health data'

        expected_collection_health_data: str = 'health_data_cons_final'

        res = self.vector_ranking.rank_collections(prompt_to_rank_health_data, 5)

        ic(res)

        self.assertEqual(res[0][0], expected_collection_health_data)

    def test_vector_ranking_alerts(self):
        prompt_to_rank_alerts: str = "Give me top alerts in last one month"

        expected_collection_alerts: str = 'alerts'

        res = self.vector_ranking.rank_collections(prompt_to_rank_alerts, 5)

        ic(res)

        self.assertEqual(res[0][0], expected_collection_alerts)


if __name__ == '__main__':
    unittest.main()
