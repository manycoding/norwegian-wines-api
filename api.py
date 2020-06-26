import os

from algoliasearch.search_client import SearchClient
from algoliasearch.exceptions import RequestException
from flask import Flask
from flask_restful import Resource, Api, reqparse
import numpy as np

app = Flask(__name__)
api = Api(app)
ALGOLIA_ID = os.environ.get("ALGOLIA_ID")
ALGOLIA_KEY = os.environ.get("ALGOLIA_KEY")
ALGOLIA_INDEX = os.environ.get("ALGOLIA_INDEX")
DEBUG = os.environ.get("DEBUG", False)


class Similar(Resource):
    def get(self, object_id: str):
        """Get similar products for object_id
        Returns:
            A list of top 100 similar object_ids
        """
        try:
            result = index.get_object(
                object_id, {"attributesToRetrieve": ["description-similar"]}
            )
        except RequestException as e:
            return f"objectID {object_id} does not exist", 404

        return (
            list(np.array(result.get("description-similar", []))[:, 0]),
            200,
        )


api.add_resource(Similar, "/similar/<string:object_id>")

if __name__ == "__main__":
    client = SearchClient.create(ALGOLIA_ID, ALGOLIA_KEY)
    index = client.init_index(ALGOLIA_INDEX)
    app.run(debug=DEBUG)
