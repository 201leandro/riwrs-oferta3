from flask import Flask, request, jsonify
from flask_cors import CORS

from functions import tweetsReport


application = Flask(__name__)
application.debug=True
cors = CORS(application, resources={r"/*": {"origins": "*"}})


@application.route("/")
def index():

    name = request.args.get("name")

    if name:
        return jsonify(tweetsReport(name))
    else:
        return jsonify({"name": "this param is required"}), 400


if __name__ == '__main__':
    application.run()
