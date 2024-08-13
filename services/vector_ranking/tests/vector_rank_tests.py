import unittest

import numpy as np
from icecream import ic

from services.vector_ranking.vector_ranking import VectorRanking


class TestVector(unittest.TestCase):

    vector_ranking = VectorRanking()

    def test_vector_ranking(self):

        prompt_to_rank = "Can you give me the data for POD106 and explain the P95 for network health data"

        expected_collection = "health_data_cons_final"

        res = self.vector_ranking.rank_collections(prompt_to_rank)

        self.assertEqual(res[0][0], expected_collection)


if __name__ == '__main__':
    unittest.main()
