from flask import Flask, request, render_template

from flask_cors import CORS


application = Flask(__name__)
application.debug=True
cors = CORS(application, resources={r"/*": {"origins": "*"}})


@application.route("/")
def index():
    return render_template("content.html")


if __name__ == '__main__':
    application.run(port=4000)

# application.run(port=4000)