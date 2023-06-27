from app import plate_utils
import os

if __name__ == '__main__':
    source = '../../13518006/TA/ANPR/examples/test_video_1.mp4'
    print("file exists?", os.path.exists(source))
    plate_utils.get_plates_from_video(source)
    # plate_utils.get_plates_from_video(0)
