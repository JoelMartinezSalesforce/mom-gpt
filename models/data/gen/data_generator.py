import os
import json
import random
import string

from tqdm import tqdm


class MockDataGenerator:
    def __init__(self, data_struct):
        self.data_struct = data_struct
        self.data_path = '../dump/'
        self.data_exists = os.path.exists(self.data_path)

        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    @staticmethod
    def _generate_random_string(length=10):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def _generate_random_number(start=0, end=100):
        return random.uniform(start, end)

    @staticmethod
    def _generate_random_int(start=0, end=1000):
        return random.randint(start, end)

    def _generate_random_data(self, structure):
        random_data = {}
        for key, value in structure.items():
            if isinstance(value, dict):
                random_data[key] = self._generate_random_data(value)
            elif isinstance(value, str):
                if "interface" in key or "device" in key or "site" in key or "alias" in key or "type" in key or "util-datasource" in key:
                    random_data[key] = self._generate_random_string()
                elif "json" in key:
                    random_data[key] = json.dumps({
                        str(self._generate_random_int(700, 800)): {
                            "site": self._generate_random_string(),
                            "device": self._generate_random_string(),
                            "interface": self._generate_random_string()
                        }
                    })
                elif "status" in key:
                    random_data[key] = random.choice(['active', 'inactive'])
                elif "speed" in key:
                    random_data[key] = self._generate_random_int(1000000000, 100000000000)
                elif "percent" in key:
                    random_data[key] = self._generate_random_number(0, 1)
                elif "receive" in key or "transmit" in key or "max" in key:
                    random_data[key] = self._generate_random_number(0, 1000)
                elif "isis-metric" in key or "if-index" in key:
                    random_data[key] = random.choice([self._generate_random_int(1, 100), None])
                elif "util-updated" in key:
                    random_data[key] = self._generate_random_int(1609459200000,
                                                                 1735689600000)  
                elif "edge-latency" in key:
                    random_data[key] = random.choice([self._generate_random_number(0, 100), None])
                elif "edge-side" in key:
                    random_data[key] = random.choice([self._generate_random_string(), None])
                else:
                    random_data[key] = value
            elif isinstance(value, (int, float)):
                random_data[key] = self._generate_random_number() if isinstance(value,
                                                                                float) else self._generate_random_int()
            else:
                random_data[key] = value
        return random_data

    def create_new_dump(self, n: int = 5) -> None:
        """
        Creates a new data dump meaning it will take as an argument the number of records this should be creating
        :param n: number of records to create
        :return None:
        """
        print("Creating a new Dump of data")

        data = [self._generate_random_data(self.data_struct) for _ in tqdm(range(n), desc="Generating data")]
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
