from app import plate_utils
import os

if __name__ == '__main__':
    print("file exists?", os.path.exists('../../../13518006/TA/ANPR/examples/test_video_1.mp4'))
    plate_utils.get_plates_from_video('../../../13518006/TA/ANPR/examples/test_video_2.mp4')
    # plate_utils.get_plates_from_video(0)
