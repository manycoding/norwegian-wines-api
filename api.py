import os
from pathlib import Path
import zipfile

from algoliasearch.search_client import SearchClient
from algoliasearch.exceptions import RequestException
import boto3
from flask import Flask
from flask_restful import Resource, Api
import numpy as np
import scipy
from sentence_transformers import SentenceTransformer
from webargs import fields, validate
from webargs.flaskparser import use_args, use_kwargs, parser, abort


app = Flask(__name__)
api = Api(app)
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


class Similar(Resource):
    @use_kwargs(
        {"object_ids": fields.List(fields.Str(), required=True)}, location="json",
    )
    def post(self, object_ids):
        """Get similar products for object_ids
        Returns:
            A list of top 100 similar object_ids
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


@parser.error_handler
def handle_request_parsing_error(err, req, schema, *, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(error_status_code, errors=err.messages)


api.add_resource(Similar, "/similar")

if __name__ == "__main__":
    app.run(debug=DEBUG)
