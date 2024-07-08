import os
import json
import random
import string


class MockDataGenerator:
    def __init__(self):
        self.data_struct = {
            "remote": {
                "device": " ",
                "interface": " ",
                "site": " ",
            },
            "logical-indexes-json": "{\"702\":{\"site\":\"fra2\",\"device\":\"bbr10-fra5\","
                                    "\"interface\":\"xe-0/0/11.0\"}}",
            "interface": "xe-0/0/11.0",
            "site": "fra2",
            "collection-site": "fra",
            "speed": 10000000000,
            "util-updated": 1719433741046,
            "type": "edge-interface",
            "edge-latency": None,
            "receive": 936.380357233664,
            "if-index": "345",
            "alias": " OnPremDDOS-M",
            "status": "active",
            "max": 936.380357233664,
            "transmit": 106.8903065674496,
            "device": "bbr100-fra50",
            "isis-metric": None,
            "percent": 0.00009363803572,
            "edge-side": "aloc",
            "util-datasource": "ArgusLOL",
            "edge-speed": "",
        }

        self.data_path = '../dump/'
        self.data_exists = os.path.exists(self.data_path)

        # Ensure the data path exists
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    @staticmethod
    def _generate_random_string(length=10):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def _generate_random_data(self):
        random_data = self.data_struct.copy()

        # Generate random values for the fields, allowing some fields to be null
        random_data['remote'] = {
            "device": self._generate_random_string() if random.choice([True, False]) else None,
            "interface": self._generate_random_string() if random.choice([True, False]) else None,
            "site": self._generate_random_string() if random.choice([True, False]) else None,
        } if random.choice([True, False]) else None

        random_data['logical-indexes-json'] = json.dumps({
            str(random.randint(700, 800)): {
                "site": self._generate_random_string(),
                "device": self._generate_random_string(),
                "interface": self._generate_random_string()
            }
        })
        random_data['interface'] = self._generate_random_string()
        random_data['site'] = self._generate_random_string()
        random_data['collection-site'] = self._generate_random_string()
        random_data['speed'] = random.randint(1000000000, 100000000000)
        random_data['util-updated'] = random.randint(1609459200000, 1735689600000)  # Timestamp range for 2021-2024
        random_data['type'] = self._generate_random_string()
        random_data['edge-latency'] = random.choice([random.uniform(0, 100), None])
        random_data['receive'] = random.uniform(0, 1000)
        random_data['if-index'] = str(random.randint(200, 1000))
        random_data['alias'] = self._generate_random_string()
        random_data['status'] = random.choice(['active', 'inactive'])
        random_data['max'] = random.uniform(0, 1000)
        random_data['transmit'] = random.uniform(0, 1000)
        random_data['device'] = self._generate_random_string()
        random_data['isis-metric'] = random.choice([random.randint(1, 100), None])
        random_data['percent'] = random.uniform(0, 1)
        random_data['edge-side'] = random.choice([self._generate_random_string(), None])
        random_data['util-datasource'] = self._generate_random_string()
        random_data['edge-speed'] = random.choice([random.randint(1000000000, 100000000000), None])

        return random_data

    def create_new_dump(self, n: int = 5) -> None:
        """
        Creates a new data dump meaning it will take as an argument the number of records this should be creating
        :param n: number of records to create
        :return None:
        """
        print("Creating a new Dump of data")

        data = [self._generate_random_data() for _ in range(n)]
        file_path = os.path.join(self.data_path, 'data_dump.json')

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Data dump created with {n} records.")

    def reset_data_dump(self) -> None:
        """
        Resets the data dump meaning deletes the previous data json collection and does nothing afterward
        :return None:
        """
        print("Resetting data dump")

        file_path = os.path.join(self.data_path, 'data_dump.json')

        if os.path.exists(file_path):
            os.remove(file_path)
            print("Data dump reset successfully.")
        else:
            print("No data dump found to reset.")

    def recall_data_dump(self) -> list:
        """
        Checks if the data dump exists and if it does, recreates the data dump returning the file as a json
        otherwise
        :return: list with the data dump
        """
        file_path = os.path.join(self.data_path, 'data_dump.json')

        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            print("Data dump recalled successfully.")
            return data
        else:
            print("No data dump found.")
            return []
