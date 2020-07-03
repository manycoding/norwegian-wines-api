import os
from pathlib import Path
import zipfile
import re
from typing import *

from algoliasearch.search_client import SearchClient
from algoliasearch.exceptions import RequestException
import boto3
from flask import Flask
from flask_restful import Resource, Api
import numpy as np
import pandas as pd
import scipy
from sentence_transformers import SentenceTransformer
from webargs import fields, validate
from webargs.flaskparser import parser, abort
from flask_apispec import FlaskApiSpec, use_kwargs, marshal_with, MethodResource


app = Flask(__name__)
api = Api(app)
docs = FlaskApiSpec(app)
ALGOLIA_ID = os.environ.get("ALGOLIA_ID")
ALGOLIA_KEY = os.environ.get("ALGOLIA_KEY")
ALGOLIA_INDEX = os.environ.get("ALGOLIA_INDEX")
DEBUG = os.environ.get("DEBUG", False)
client = SearchClient.create(ALGOLIA_ID, ALGOLIA_KEY)
index = client.init_index(ALGOLIA_INDEX)


def get_embedder():
    model_path = Path("/tmp/multilingual-no")
    zip_path = Path("/tmp/multilingual-no.zip")
    s3_client = boto3.client("s3")
    bucket = "norwegian-wines-models"

    if not zip_path.exists():
        s3_client.download_file(
            bucket, "multilingual-no-20200628T162921Z-001.zip", str(zip_path),
        )

    embeddings_path = Path("/tmp/no_embeddings.npy")
    if not embeddings_path.exists():
        s3_client.download_file(bucket, "no_embeddings.npy", str(embeddings_path))
    embeddings = np.load(embeddings_path, allow_pickle=True)

    if not model_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("/tmp/")
    return SentenceTransformer(str(model_path)), embeddings


embedder, corpus_embeddings = get_embedder()


class Similar(MethodResource):
    @use_kwargs(
        {"object_ids": fields.List(fields.Str(), required=True)}, locations=["json"],
    )
    def post(self, object_ids: List[str]) -> Tuple[Dict, int]:
        """Get similar products for object_ids
        Returns:
            A tuple of dict with message and results, and a status code.
            Results contain a list of top 100 similar object_ids
        """
        answer = {}
        if not object_ids:
            answer["message"] = "'object_ids' should not be empty"
            return answer
        if len(object_ids) == 1:
            result, response = get_attribute(object_ids, "description-similar")
            if result:
                answer["results"] = list(np.array(result[0])[:, 0])
            if response.get("message"):
                answer["message"] = response.get("message")
                answer["results"] = result
            return (
                answer,
                200,
            )
        elif len(object_ids) > 1:
            result, response = get_attribute(object_ids, "description-profile")
            query = ". ".join(result)
            if response.get("message"):
                answer["message"] = response.get("message")
            if not query:
                answer[
                    "message"
                ] += f"{', '.join(object_ids)} have no 'description-profile'"
                return answer, 404
            query_embeddings = embedder.encode([query])
            distances = scipy.spatial.distance.cdist(
                query_embeddings, list(corpus_embeddings[:, 1]), "cosine"
            )[0]
            results = zip(corpus_embeddings[:, 0], distances)
            results = sorted(results, key=lambda x: x[1])
            results = list(filter(lambda x: x[0] not in object_ids, results))[:100]
            answer["results"] = list(np.array(results)[:, 0])
            return answer, 200


def get_attribute(object_ids, attr):
    try:
        result = index.get_objects(object_ids, {"attributesToRetrieve": [attr]})
        return [d[attr] for d in result["results"] if d], result
    except RequestException as e:
        return [], result


class Best(MethodResource):
    index.set_settings({"attributesForFaceting": ["filterOnly(rating-price-rank)"]})

    @use_kwargs(
        {"n": fields.Int(missing=99)}, locations=["json"],
    )
    def post(self, n: int) -> Tuple[Dict, int]:
        """Get best wines by rating to pricePerLiter rank
        Returns:
            A tuple of dict with message and results, and a status code.
            Results contain a list of best wine names with object_ids
        """
        answer = {}
        response = get_products_with(f"rating-price-rank: 0 TO {n}")
        response = sorted(response, key=lambda x: x["rating-price-rank"])
        answer["results"] = [
            {"name": r["name"], "objectID": r["objectID"]} for r in response
        ]
        return answer, 200


def get_products_with(filters: str = "") -> List[Dict]:
    results = []
    res = index.browse_objects({"filters": filters})
    while True:
        try:
            results.append(res.next())
        except:
            break
    return results


class BestCountries(MethodResource):
    @use_kwargs(
        {"min_total": fields.Int(missing=1)}, locations=["json"],
    )
    def post(self, min_total: int) -> Tuple[Dict, int]:
        """Get countries with average rating
        """
        answer = {}
        response = get_products_with()
        wines = pd.DataFrame(response).replace("", np.nan)
        countries = pd.DataFrame(
            wines["origins-origin-country"].value_counts(dropna=True)
        ).sort_index()
        countries.rename(columns={"origins-origin-country": "total"}, inplace=True)
        countries["average_rating"] = (
            wines.groupby(["origins-origin-country"])["aggregateRating-ratingValue"]
            .mean()
            .sort_index()
        )
        countries.sort_values(["average_rating"], ascending=False, inplace=True)
        answer["results"] = countries.round(4)[countries["total"] > min_total].to_dict(
            "index"
        )
        return answer, 200


class BestRegions(MethodResource):
    @use_kwargs(
        {"min_total": fields.Int(missing=10), "min_rating": fields.Float(missing=4.0)},
        locations=["json"],
    )
    def post(self, min_total: int, min_rating: float) -> Tuple[Dict, int]:
        """Get regions with percentage of wines with good rating
        """
        answer = {}
        response = get_products_with()
        wines = (
            pd.DataFrame(response)
            .replace(["", 0], np.nan)
            .dropna(subset=["aggregateRating-ratingValue", "origins-origin-region"])
        )
        regions = pd.DataFrame(
            wines["origins-origin-region"].value_counts()
        ).sort_index()
        regions.rename(columns={"origins-origin-region": "total"}, inplace=True)
        regions_group = wines.groupby("origins-origin-region")[
            "aggregateRating-ratingValue"
        ]
        regions["top_wines_percentage"] = regions_group.apply(
            lambda x: len(x[x >= min_rating]) / len(x) * 100
        ).round()
        regions = regions.sort_values(
            ["top_wines_percentage", "total"], ascending=False
        )

        answer["results"] = regions[regions["total"] > min_total].to_dict("index")
        return answer, 200


@parser.error_handler
def handle_request_parsing_error(err, req, schema, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(error_status_code, errors=err.messages)


resources = [Similar, Best, BestCountries, BestRegions]
for r in resources:
    url = re.sub("(?!^)([A-Z]+)", r"_\1", r.__name__).lower()
    api.add_resource(r, f"/{url}")
    docs.register(r)

if __name__ == "__main__":
    app.run(debug=DEBUG)
