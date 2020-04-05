from flask import request
from flask.views import MethodView

from application.digitalization_photo_service.digitalization_service import ColorizationService
from application.rest.request_mapper.colorize_request import ColorizeRequest


class ColorizeFaceResource(MethodView):
    def __init__(self):
        self._colorize_self_service = ColorizationService()

    def post(self):
        face = ColorizeRequest(request.get_json()['base64String'], request.get_json()['chatId'])
        base64_string_colorized = self._colorize_self_service.colorize(face)
        return {'base64String': base64_string_colorized, 'chatId': face.chat_id}
