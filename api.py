import os

from algoliasearch.search_client import SearchClient
from algoliasearch.exceptions import RequestException
from flask import Flask
from flask_restful import Resource, Api
import numpy as np
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


class Similar(Resource):
    @use_kwargs(
        {"object_ids": fields.List(fields.Str(), required=True)}, location="json",
    )
    def post(self, object_ids):
        """Get similar products for object_id
        Returns:
            A list of top 100 similar object_ids
        """
        if len(object_ids) == 1:
            try:
                result = index.get_object(
                    object_ids[0], {"attributesToRetrieve": ["description-similar"]}
                )
            except RequestException as e:
                return f"objectID {object_ids[0]} does not exist", 404

            return (
                list(np.array(result.get("description-similar", []))[:, 0]),
                200,
            )


@parser.error_handler
def handle_request_parsing_error(err, req, schema, *, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(error_status_code, errors=err.messages)


api.add_resource(Similar, "/similar")

if __name__ == "__main__":
    app.run(debug=DEBUG)
