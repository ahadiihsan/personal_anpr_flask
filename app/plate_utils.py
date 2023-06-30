
from ultralytics import YOLO
from copy import deepcopy
from app import utils
import cv2
import torch
import math
import numpy as np
import keras_ocr
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import easyocr
import Levenshtein
from app import save_image

# Load detection model
model = YOLO('./result/train4/weights/best.pt')

# load recognition model
recognizer = keras_ocr.recognition.Recognizer()
ocr_model = recognizer.model.load_weights('./models/plat1.h5')
reader1 = keras_ocr.pipeline.Pipeline(recognizer=ocr_model)
reader2 = easyocr.Reader(['en'], gpu=True)


def detect_plate(source_image):
    # with torch.no_grad():
    # Inference
    pred = model(
            source_image,
            augment=True,
            iou=0.7,
            half=True,
            device='mps',
            conf=0.7,
            agnostic_nms=True,
        )[0]

    plate_detections = []
    det_confidences = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Return results
            for box in det.boxes:
                coords = [
                        int(position) for position in (
                            box.xyxy.clone().detach().view(1,4)
                            # torch.tensor(box.xyxy).view(1, 4)
                        ).tolist()[0]
                    ]
                plate_detections.append(coords)
                det_confidences.append(box.conf)
                print(f"DET CONFIDENCE: {box.conf}")
    
    return plate_detections, det_confidences


def extract_plate(image, coord):
    h, w, c = image.shape
    # minus
    x1 = int(coord[1])-10 if int(coord[1])-10 >= 0 else int(coord[1])
    y1 = int(coord[0]) if int(coord[0]) >= 0 else int(coord[0])
    # plus
    x2 = int(coord[3])+10 if int(coord[3])+10 <= w else int(coord[3])
    y2 = int(coord[2]) if int(coord[2]) <= h else int(coord[2])

    cropped_image = image[
            x1:x2,
            y1:y2
        ]

    if save_image: cv2.imwrite("./image/img_cropped.jpeg", cropped_image)

    return cropped_image


def recognition_preprocessing_1(input):

    plate_image = input

    plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    if save_image: cv2.imwrite("./image/warp/imgGray.jpeg", plate_image)
    
    plate_image = utils.rotate(plate_image)
    if save_image: cv2.imwrite("./image/img_rotated.jpeg", plate_image)

    plate_image = utils.maximizeContrast(plate_image)
    if save_image: cv2.imwrite("./image/img_contrast.jpeg", plate_image)
    
    plate_image = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("./image/img_color.jpeg", plate_image)

    plate_image = cv2.resize(
                                plate_image,
                                None,
                                fx=2,
                                fy=1,
                                interpolation=cv2.INTER_CUBIC
                            )
    if save_image: cv2.imwrite("./image/img_rescaled.jpeg", plate_image)

    return plate_image


def recognition_preprocessing_2(input):

    kernel = np.ones((3, 3))
    plate_image = input

    plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    if save_image: cv2.imwrite("./image/warp/imgGray.jpeg", plate_image)

    imgCanny = cv2.Canny(plate_image, 50, 150)
    if save_image: cv2.imwrite("./image/warp/imgCanny.jpeg", imgCanny)

    imgDilate = cv2.dilate(imgCanny, kernel, iterations=2)
    if save_image: cv2.imwrite("./image/warp/imgDilate.jpeg", imgDilate)

    imgThres = cv2.erode(imgDilate, kernel, iterations=2)
    if save_image: cv2.imwrite("./image/warp/imgThres.jpeg", imgThres)

    biggest, imgContour, warped = utils.getContours(imgThres, input)

    if save_image: cv2.imwrite("./image/warp/imgContour.jpeg", imgContour)
    if save_image: cv2.imwrite("./image/img_warped.jpeg", warped)

    plate_image = warped

    plate_image = cv2.resize(
                                plate_image,
                                None,
                                fx=2,
                                fy=1,
                                interpolation=cv2.INTER_CUBIC
                            )
    if save_image: cv2.imwrite("./image/img_rescaled.jpeg", plate_image)
    
    # plate_image = cv2.bitwise_not(plate_image)
    # if save_image: cv2.imwrite("./image/img_not.jpeg", plate_image)

    return plate_image


def get_distance(predictions):
    """
    Function returns list of dictionaries with (key,value):
        * text : detected text in image
        * center_x : center of bounding box (x)
        * center_y : center of bounding box (y)
        * distance_from_origin : hypotenuse
        * distance_y : distance between y and origin (0,0)
    ...for each bounding box (detections).
    """

    # Point of origin
    x0, y0 = 0, 0

    # Generate dictionary
    detections = []
    for group in predictions:

        # Get center point of bounding box
        top_left_x, top_left_y = group[1][0]
        bottom_right_x, bottom_right_y = group[1][1]
        center_x, center_y = (
                (top_left_x + bottom_right_x)/2,
                (top_left_y + bottom_right_y)/2
            )

        # Use the Pythagorean Theorem to solve for distance from origin
        distance_from_origin = math.dist([x0, y0], [center_x, center_y])

        # Calculate difference between y and origin to get unique rows
        distance_y = center_y - y0

        # Append all results
        detections.append({
                            'text': group[0],
                            'center_x': center_x,
                            'center_y': center_y,
                            'distance_from_origin': distance_from_origin,
                            'distance_y': distance_y
                        })

    return detections


def distinguish_rows(lst, thresh=15):
    """Function to help distinguish unique rows"""
    sublists = []
    for i in range(0, len(lst)-1):
        if (lst[i+1]['distance_y'] - lst[i]['distance_y'] <= thresh):
            if lst[i] not in sublists:
                sublists.append(lst[i])
            sublists.append(lst[i+1])
        else:
            yield sublists
            sublists = [lst[i+1]]
    yield sublists


def ocr_plate1(plate_image):
    result = ""

    predict = reader1.recognize([plate_image])

    predictions = get_distance(predict[0])
    predictions = list(distinguish_rows(predictions, 10))

    # Remove all empty rows
    predictions = list(filter(lambda x: x != [], predictions))

    # Order text detections in human readable format
    ordered_preds = []

    for row in predictions:
        row = sorted(row, key=lambda x: x['distance_from_origin'])
        for each in row:
            ordered_preds.append(each['text'])

    for data in ordered_preds[:3]:
        result += data.upper()

    print(f"OCR RESULT: {result}")
    return result, 0


def ocr_plate2(plate_image):
    result = ""
    predictions = reader2.readtext(plate_image)
    i = 0
    confidence = 0
    for row in predictions:
        if row[2] > 0.7:
            result += row[1]
            i += 1
            confidence += row[2]
        if i == 3:
            break

    if confidence != 0:
        confidence = confidence/3

    return result.replace(" ", ""), confidence


def get_best_ocr1(preds, rec_conf, ocr_res, track_id):
    for info in preds:
        # Check if it is current track id
        if info['track_id'] == track_id:
            # Check if the ocr confidenence is maximum or not
            if info['ocr_conf'] < rec_conf:
                info['ocr_conf'] = rec_conf
                info['ocr_txt'] = ocr_res
            else:
                rec_conf = info['ocr_conf']
                ocr_res = info['ocr_txt']
            break
    return preds, rec_conf, ocr_res

def get_best_ocr2(preds, rec_conf, ocr_res, track_id):
    for info in preds:
        # Check if it is current track id
        if info['track_id'] == track_id:
            info['ocr_conf'] = rec_conf
            info['ocr_txt'] = ocr_res
    return preds, rec_conf, ocr_res


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = (
            line_thickness or
            round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
         )  # line/font thickness
    color = color or [math.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA
            )


def get_plates_from_image(input, filename=""):

    if input is None:
        return None

    plate_detections, det_confidences = detect_plate(input)

    plate_texts = []
    ocr_confidences = []
    plate_text = ''
    detected_image = deepcopy(input)

    for coords in plate_detections:
        plate_image = extract_plate(input, coords)
        
        to_ocr = recognition_preprocessing_2(plate_image)
        plate_text, ocr_confidence = ocr_plate1(to_ocr)
        if len(plate_text) <= 3:
            to_ocr = recognition_preprocessing_1(plate_image)
            plate_text, ocr_confidence = ocr_plate1(to_ocr)

        plate_texts.append(plate_text)
        ocr_confidences.append(ocr_confidence)

        plot_one_box(
                coords,
                detected_image,
                label=plate_text,
                color=[0, 150, 255],
                line_thickness=2
            )

        if filename != "":
            path = './image/results/'+filename+'/'

            if not os.path.exists(path):
                os.mkdir(path)

            if save_image: cv2.imwrite(
                "{path}{idx}.jpeg".format(
                    path=path,
                    idx=filename.split("_")[1]
                    ),
                detected_image
            )

            if save_image: cv2.imwrite(
                "{path}img_cropped_{plate}.jpeg".format(
                        path=path,
                        plate=plate_text
                    ),
                plate_image
            )
            if save_image: cv2.imwrite(
                "{path}img_to_ocr_{plate}.jpeg".format(
                        path=path,
                        plate=plate_text
                    ),
                to_ocr
            )

        if save_image: cv2.imwrite("./image/img_recognized.jpeg", detected_image)

    return detected_image, plate_text


def get_plates_from_video(source):
    if source is None:
        return None

    # Create a VideoCapture object
    video = cv2.VideoCapture(source)

    # Intializing tracker
    tracker = DeepSort(embedder_gpu=False)
    
    # Initialize fps variables
    frame_count = 0
    start_time = time.time()

    # Initializing some helper variables.
    preds = []
    idx = 0
    
    # Initializing results variables
    rows = []
    save_count = 0
    
    while (True):
        ret, frame = video.read()
        if ret:
            # Run the ANPR algorithm
            bboxes, scores = detect_plate(frame)
            # Convert Pascal VOC detections to COCO
            bboxes = list(map(
                                lambda bbox: utils.pascal_voc_to_coco(bbox),
                                bboxes
                            ))

            if len(bboxes) > 0:
                # Storing all the required info in a list.
                detections = [
                        (bbox, score, 'number_plate')
                        for bbox, score in zip(bboxes, scores)
                    ]

                # Applying tracker.
                # The tracker code flow: kalman filter -> target association
                # (using hungarian algorithm) and appearance descriptor.
                tracks = tracker.update_tracks(detections, frame=frame)

                # Checking if tracks exist.
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    # Changing track bbox to top left, bottom right coordinates
                    bbox = [
                            int(position)
                            for position in list(track.to_tlbr())
                        ]

                    for i in range(len(bbox)):
                        if bbox[i] < 0:
                            bbox[i] = 0
                    
                    # only consider bboxes that are on the detection range
                    if bbox[0] < 100 or bbox[2] > 1200 or bbox[1] < 300 or bbox[3] > 500:
                        continue
                    
                    # Cropping the license plate and applying the OCR.
                    plate_image = extract_plate(frame, bbox)
                    
                    # recognize plate text
                    to_ocr = recognition_preprocessing_2(plate_image)
                    plate_text, ocr_confidence = ocr_plate1(to_ocr)
                    if len(plate_text) <= 3:
                        to_ocr = recognition_preprocessing_1(plate_image)
                        plate_text, ocr_confidence = ocr_plate1(to_ocr)

                    if ocr_confidence == 0:
                        ocr_confidence = scores[0].cpu().numpy()[0]
                    # Storing the ocr output for corresponding track id.
                    output_frame = {
                        'track_id': track.track_id,
                        'ocr_txt': plate_text,
                        'ocr_conf': ocr_confidence
                        }

                    # Appending track_id to list
                    # only if it does not exist in the list
                    # else looking for the current track in the list and
                    # updating the highest confidence of it.
                    if track.track_id not in list(
                                set(pred['track_id'] for pred in preds)
                            ):
                        preds.append(output_frame)
                        save_count = 0
                    else:
                        save_count +=1
                        preds, ocr_confidence, plate_text = get_best_ocr2(
                                preds,
                                ocr_confidence,
                                plate_text,
                                track.track_id
                            )
                    
                    if save_count <= 5:
                        row = [track.track_id, plate_text, ocr_confidence]
                        rows.append(row)
                    
                    # Plotting the prediction.
                    plot_one_box(
                            bbox,
                            frame,
                            label=f'{str(track.track_id)}. {plate_text}',
                            color=[255, 150, 0],
                            line_thickness=3
                        )

                    if save_image: cv2.imwrite(
                        "./image/img_recognized{idx}.jpeg".format(idx=idx),
                        frame
                        )
                    idx += 1
            
            frame_count += 1
            
            plot_one_box(
                            [100,300,1200,500],
                            frame,
                            label="",
                            color=[255, 150, 0],
                            line_thickness=3
                        )

            # Write the frame into the output file
            cv2.imshow("Output", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate FPS
    fps = frame_count / elapsed_time

    # When everything done, release the video capture and video write objects
    video.release()
    cv2.destroyAllWindows()
    
    print(f"FRAME COUNT: {frame_count}")
    print(f"ELAPSED TIME: {elapsed_time}")
    print(f"FPS: {fps:.2f}")

    return rows

