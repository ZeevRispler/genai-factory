import certifi
import pandas as pd
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient


def get_engine(connection_url: str) -> MongoClient:
    """
    Get the MongoDB Connection Client.

    :param connection_url: The MongoDB connection String.

    :return: The MongoDB connection String.
    """
    client = MongoClient(connection_url, tlsCAFile=certifi.where())
    return client


def get_user_card_data(
    engine: MongoClient,
    user_id: str,
) -> pd.DataFrame:
    db = engine["bfsi"]
    collection = db["user_data"]
    pipeline = [
        {"$match": {"Customer_ID": int(user_id)}},
        {
            "$project": {
                "_id": 0,
                "Occupation": 1,
                "Annual_Income": 1,
                "Credit_Mix": 1,
                "Num_Credit_Card": 1,
            }
        },
    ]

    items = list(collection.aggregate(pipeline))
    if not items:
        return pd.DataFrame()
    if items[0].get("Credit_Mix") != "Bad":
        items[0].pop("Credit_Mix")
        items[0].pop("Num_Credit_Card")
        income_class = "HIGH" if items[0]["Annual_Income"] > 50000 else "MEDIUM"
    else:
        income_class = "LOW"
    # items = [item["Credit_Score"] for item in items]
    # items = pd.DataFrame(items)

    return income_class


def get_card_description(
    engine: MongoClient,
    card_name: str = None,
    income_class: str = None,
) -> pd.DataFrame:
    db = engine["bfsi"]
    collection = db["card_info"]
    # if we have the card name, this is a user specific query, and we just look for card details
    if card_name:
        pipeline = [
            {
                "$search": {
                    "index": "search_index",
                    "text": {"query": card_name, "path": "card_name"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "image_name": 1,
                    "annual_fees": 1,
                    "joining_fees": 1,
                    # 'income_class': 1,
                }
            },
        ]
    # if we have the income class, this is a general query, and we look for the best card for the user, sort by fees
    elif income_class:
        pipeline = [
            {
                "$match": {
                    "income_class": income_class,
                }
            },
            {"$sort": {"annual_fees": 1, "joining_fees": 1}},
            {
                "$project": {
                    "_id": 0,
                    "card_name": 1,
                    # 'income_class': 1,
                    "annual_fees": 1,
                    "joining_fees": 1,
                    "image_name": 1,
                }
            },
        ]
    items = list(collection.aggregate(pipeline))
    # items = [item["Credit_Score"] for item in items]
    items = pd.DataFrame(items)

    return items


class MongoDBConnector(MongoDBAtlasVectorSearch):
    def __init__(self, collection_name: str, connection_args: dict, embedding_function):
        self._connection_string = connection_args.get("connection_string")
        self._namespace = connection_args.get("db_name") + "." + collection_name
        self._client = MongoClient(self._connection_string, tlsCAFile=certifi.where())
        self._collection = self._client[self._namespace.split(".")[0]][
            self._namespace.split(".")[1]
        ]
        self._embedding = embedding_function
        self._index_name = connection_args.get("index_name")
        self._text_key = connection_args.get("text_key", "text")
        self._embedding_key = connection_args.get("embedding_key", "embedding")
        self._relevance_score_fn = connection_args.get("relevance_score_fn", "cosine")
