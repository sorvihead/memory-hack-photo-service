from flask import jsonify
from flask import request
from flask.views import MethodView

from application.facenet_service.face_embedding import FaceEmbedding
from application.rest.request_mapper.faces_request import FacesRequest


class ComparedFacesResource(MethodView):
    def __init__(self):
        self._compared_faces_service = FaceEmbedding()

    def post(self):
        faces = request.get_json()['faces']
        print(request.get_json())
        deserialized_faces = [FacesRequest(face['chatId'],
                                           face['base64String'],
                                           face['type'])
                              for face in faces]
        similarity = self._compared_faces_service.compare(deserialized_faces)
        return jsonify({'percent': similarity})
