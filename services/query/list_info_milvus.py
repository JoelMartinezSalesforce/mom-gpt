from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility

if __name__ == '__main__':

    # Initialize connection to Milvus
    connections.connect("default", host='localhost', port='19530', user='username', password='password')

    for collection in utility.list_collections():
        print(collection)
        if not (collection == 'health_data_cons_final'):
            utility.drop_collection(collection)


