# personal_anpr_flask
Developing Automatic Number Plate Recognition for Multi Lane Free Flow Toll E-Collection using Machine Learning Approach

## Getting started

### Requirements

To start the web service you should have **Python 3.8 or higher** and **Tesseract-OCR** program installed. Tesseract-OCR program is available on all platforms and can be installed using 
official documentation: https://github.com/tesseract-ocr/tesseract. For Windows you can install program in `C:/Program Files/Tesseract-OCR/tesseract.exe` path so it's compatible with current code without the extra configuration. Linux will work out-of-the-box.

To install required Python modules using PIP, you can use the following command in the project root directory:

```bash
pip3 install -r requirements.txt
```

### Running

It's recommended to set Python virtual environment before running script or to use PyCharm program which will do this automatically. To start the web service run the following command:

```bash
python3 run.py
```

## REST API

There are two available API-s defined in the web service - one for detecting license plates from the image, and the other for checking if given license plate is detected.

### Detecting license plate from image

Web service can detect license plate from all cameras that have the ability to save images in JPG or PNG format. Testing the service could also be done using Postman or a similar program with example images from the Internet. Images like this one below will work without any problems and license plate will be extracted from the image (license plate on given image is blurred because this is a live image captured from the security camera).

<p align="center"><img src="https://github.com/SanjinKurelic/FlaskALPR/blob/master/media/LicencePlate.jpg" alt="License plate example image"/></p>

#### Request

```
POST /upload-camera-image

Content-Type: multipart/form-data
Content-Disposition: form-data; name="image"; filename="<file.jpg>"
Content-Type: image/jpeg
```

#### Repsponse

If license plate is detected:

Status code - 200 OK

```json
{
  "message": "License plate AB1234CD found",
  "status": 200
}
```

If license plate is not detected:

Status code - 200 OK

```json
{
  "message": "No license plate found",
  "status": 404
}
```

### Checking if license plate is detected

For checking if given license plate is detected from last time check was made `GET` request can be sent to `check-license-plate` URL with two variables: 

- license plate or comma separated license plates that we need to check, for example: `AB1234AB`, or `AB1234AB,CD567EF,GH888II`
- ISO datetime from the last check, for example: `2020-12-01T19:15:00`
  
The service will check if license plate is detected in the interval from given datetime and current datetime. If license plate is detected the time of detection will be filled.

#### Request

```
GET /check-license-plate/<license plate>/<iso datetime>
```

#### Response

If license plates are detected:

Status code - 200 OK

```json
[
  {
    "detected": true,
    "plate": "AB1234AB",
    "time": "2020-12-01T19:14:23"
  }
]
```

If license plates are not detected:

Status code - 200 OK

```json
[
  {
    "detected": false,
    "plate": "AB1234AB",
    "time": ""
  }
]
```

## Technologies

- Flask
- SQLAlchemy
- python-dateutil
- opencv-python-headless
- imutils
- numpy
- pytesseract
