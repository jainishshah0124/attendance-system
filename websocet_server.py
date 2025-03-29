import asyncio
import websockets
import os
import uuid
import base64
import io
from PIL import Image
import face_recognition
import cv2
import numpy as np
import JSON
import json



async def handle_websocket(websocket, path):
    print('start')
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []
    
    #retrive Data
    data=json.loads(JSON.selectJSONCALL('https://us-east-2.aws.neurelo.com/rest/employees/',"",'GET').text)["data"]
    for dtl in data:
        if(dtl["photo_path"]==''):
            continue
        img = face_recognition.load_image_file(dtl["photo_path"])
        face_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(str(dtl["employee_id"]))
    

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    while True:
        # Receive frame data from client
        frame_data = await websocket.recv()
        # Decode base64-encoded image data
        #image_data = base64.b64decode(frame_data.split(",")[1])
        _, encoded_data = frame_data.split(',', 1)
        decoded_data = base64.b64decode(encoded_data)
        
        # Convert decoded data to a NumPy array
        nparr = np.frombuffer(decoded_data, np.uint8)

        # Decode the NumPy array into an image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Only process every other frame of video to save time
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            print(name)
            face_names.append(name)
            await websocket.send(name)
            name="Unknown"

#start_server = websockets.serve(handle_websocket, "rollcallsystem.bluebush-887dce0f.eastus2.azurecontainerapps.io", 8767)
start_server = websockets.serve(handle_websocket, "127.0.0.1", 8767)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()