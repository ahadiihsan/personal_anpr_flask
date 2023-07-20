from app import models
from app import plate_utils
from app import utils
import cv2
import numpy
import time


def contains_license_plates(license_plates, date):
    # noinspection PyUnresolvedReferences
    stored_license_plates = models.LicencePlate.query.filter(
        models.LicencePlate.time >= date
    ).all()
    resp = []
    for license_plate in license_plates:
        plate = next(
                (x for x in stored_license_plates if x.plate == license_plate),
                None
            )
        # Add date check
        resp.append({
            "plate": license_plate,
            "detected": plate is not None,
            "time": (utils.convert_timestamp_to_iso(plate.time)
                    if plate is not None
                    else "")
        })

    return resp


def parse_image(image):
    if image is None:
        return None

    input = cv2.imdecode(
            numpy.fromstring(image, numpy.uint8),
            cv2.IMREAD_UNCHANGED
        )

    detected_image, plate_text = plate_utils.get_plates_from_image(
                                                           input
                                                        )

    return plate_text


def save_license_plate(plate):
    license_plate = models.LicencePlate(plate=plate, time=int(time.time()))
    license_plate.save()
