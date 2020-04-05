from flask import Flask

from application.rest.colorize_face import ColorizeFaceResource
from application.rest.compared_faces import ComparedFacesResource


def create_routes(app: Flask):
    app.add_url_rule('/compared-faces', view_func=ComparedFacesResource.as_view('compared-faces'))
    app.add_url_rule('/colorized-face', view_func=ColorizeFaceResource.as_view('colorized-face'))


def create_app() -> Flask:
    app = Flask(__name__)
    create_routes(app)
    return app
