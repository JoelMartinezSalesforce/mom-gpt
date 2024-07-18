from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    model,
)

if __name__ == '__main__':

    # Connect to Milvus
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    print("Collections in the system:", utility.list_collections())

    # Print schema and list of collections
    utility.drop_collection("my_custom_collection")
    print("Collections in the system:", utility.list_collections())
