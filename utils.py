import json
import base64


def encode_params(dictionary):
    json_string = json.dumps(dictionary)
    base64_bytes = base64.urlsafe_b64encode(json_string.encode('utf-8'))

    return base64_bytes.decode('utf-8')


def decode_params(base64_string):
    json_bytes = base64.urlsafe_b64decode(base64_string.encode('utf-8'))
    json_string = json_bytes.decode('utf-8')

    return json.loads(json_string)
