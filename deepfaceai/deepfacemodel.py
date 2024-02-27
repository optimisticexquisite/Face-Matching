from deepface import DeepFace
import os
import cv2
from preprocessing import preprocess_image

# Specify the path to the video file
video_path = 'testpv/test/Chitra/chr.mp4'

# Specify the directory to save frames
frames_dir = 'testpv/test/Chitra/frames'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Capture the video
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    # Read a new frame
    success, frame = cap.read()
    if not success:
        break  # If no frame is read, break the loop
    
    # Save the frame as an image file
    frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_path, frame)
    
    frame_count += 1

# Release the video capture object
cap.release()

img1_path = ('testpv/test/Chitra/chrp.jpg')
img2_path_folder = frames_dir
resultarray = []
def deepfacemodel(img1_path, img2_path):
    img2 = cv2.imread(img2_path)
    img2 = preprocess_image(img2)
    cv2.imwrite(img2_path, img2)
    result = DeepFace.verify(img1_path, img2_path, enforce_detection=False)
    print("Is verified: ", result["verified"])
    print("Distance: ", result["distance"])
    resultarray.append(result["verified"])
    return result

for i in os.listdir(img2_path_folder):
    try:
        deepfacemodel(img1_path, os.path.join(img2_path_folder, i))
    except Exception as e:
        print(e)
        resultarray.append("error")
        pass
print(resultarray)