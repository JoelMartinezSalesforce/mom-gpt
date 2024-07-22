from pymilvus import MilvusClient, DataType

if __name__ == '__main__':

    # 1. Set up a Milvus client
    client = MilvusClient(
        uri="http://localhost:19530"
    )

    # 2. Create a collection in quick setup mode
    client.create_collection(
        collection_name="quick_setup",
        dimension=5
    )

    res = client.get_load_state(
        collection_name="quick_setup"
    )

    print(res)

    # Output
    #
    # {
    #     "state": "<LoadState: Loaded>"
    # }

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

    schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="my_id",
        index_type="STL_SORT"
    )

    index_params.add_index(
        field_name="my_vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )

    res = client.describe_collection(
        collection_name="health_data"
    )

    print(res)

