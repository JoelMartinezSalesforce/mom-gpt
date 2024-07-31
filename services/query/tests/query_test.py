from services.query.main.query_main import Query

if __name__ == '__main__':
    query = Query("health_data_cons_final")
    res = query.ingest()
    print(res)

