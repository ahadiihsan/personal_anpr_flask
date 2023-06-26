from app import models
from app import plate_utils
from app import utils
import cv2
import numpy
import time


def contains_licence_plates(licence_plates, date):
    # noinspection PyUnresolvedReferences
    stored_licence_plates = models.LicencePlate.query.filter(
        models.LicencePlate.time >= date
    ).all()
    resp = []
    for licence_plate in licence_plates:
        plate = next(
                (x for x in stored_licence_plates if x.plate == licence_plate),
                None
            )
        # Add date check
        resp.append({
            "plate": licence_plate,
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


def save_licence_plate(plate):
    licence_plate = models.LicencePlate(plate=plate, time=int(time.time()))
    licence_plate.save()
