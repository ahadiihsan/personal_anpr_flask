from flask import Flask

app = Flask(__name__)
config = {
  "save_image": True,
  "onnx": True,
  "ocr": "easy",
}


from app import database
from app import utils
from app import models
from app import services
from app import views
from app import error_handlers