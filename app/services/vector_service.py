def classify_movie_logic(db, query_vector):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "gemini_vector_index",
                "path": "fullplot_gemini_embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": 5
            }
        },
        {"$project": {"genres": 1}}
    ]
    results = list(db.movies.aggregate(pipeline))
    # Logic đếm nhãn thể loại xuất hiện nhiều nhất ở đây...