from flask import Flask,render_template,request,redirect,url_for
from connection import engine
import sqlalchemy
import os
import face_recognition
import base64
import cv2
import numpy as np
import json
from google.cloud import storage
import io


app = Flask(__name__,template_folder=os.getcwd()+'/Templates')

@app.route('/checkConn')
def checkConn():
    try:
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text("SELECT NOW();"))
            current_time = [row[0] for row in result][0]
        return f"Database connected successfully. Current DB time: {current_time}"
    except Exception as e:
        return f"Error connecting to database: {e}"

@app.route('/register')
def register():
    return render_template('registration.html')

@app.route('/imageCapture',methods=['POST'])
def imageCapture():
    try:
        data = {
            "StudentID": int(request.form['employee_id']),
            "fname": request.form['first_name'],
            "Lname": request.form['last_name'],
            "DOB": request.form['dob'],
            "gender": request.form['gender'],
            "phone": request.form['phone'],
            "address": request.form['address'],
            "email": request.form['email'],
            "photo_path": ""  # You can fill this in later
        }

        print(data)

        insert_query = sqlalchemy.text("""
            INSERT INTO Student (StudentID, fname, Lname, DOB, gender, phone, address, email, photo_path)
            VALUES (:StudentID, :fname, :Lname, :DOB, :gender, :phone, :address, :email, :photo_path)
        """)

        with engine.connect() as conn:
            conn.execute(insert_query, data)
            print("data inserted")
            conn.commit()

        return render_template('capture.html',employee_id=request.form['employee_id'])

    except Exception as e:
        return render_template('registration.html',error=str(e))
    
@app.route('/save_image',methods=['POST','GET'])
def save_image():
    image_data = request.form['imageData']
    employee_id = request.form['employee_id']

    # Decode base64 image (remove data URL prefix)
    _, encoded_data = image_data.split(',', 1)
    decoded_data = base64.b64decode(encoded_data)

    # Upload directly to GCS
    bucket_name = 'attendance-storage'
    destination_blob_name = f'{employee_id}.jpg'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Delete existing blob if it exists
    if blob.exists():
        blob.delete()

    # Upload the image from memory
    blob.upload_from_string(decoded_data, content_type='image/jpeg')

    # Save GCS path in DB (or use blob.public_url if public)
    photo_path = f'gs://{bucket_name}/{destination_blob_name}'

    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text("UPDATE Student SET photo_path=:path WHERE StudentID=:sid"),
            {"path": photo_path, "sid": int(employee_id)}
        )
        conn.commit()

    return redirect('/attendance')

@app.route('/add_class',methods=['POST'])
def add_class():
    start_time = request.form['Time']
    subject_code = request.form['newClassName']

    try:
        with engine.connect() as conn:
            insert_query = sqlalchemy.text("""
                INSERT INTO class (start_time, subject_code)
                VALUES (:start_time, :subject_code)
            """)
            conn.execute(insert_query, {"start_time": start_time, "subject_code": subject_code})
            conn.commit()

        return redirect(url_for('attendance'))

    except Exception as e:
        # Optional: log the error or flash a message
        return f"Error adding class: {str(e)}", 500



@app.route('/attendance')
def attendance():
    localStorage_data={
         "attendanceData": '[]',
        "classes": '["CPSC 597"]',
        # "colors": '{}',
        # "debug": "honey:core-sdk:*",
        "students": json.dumps({"CPSC 597":[{"name":"Jainish Shah","rollNumber":"123"},{"name":"Nandish Amin","rollNumber":"9876"}]}),
        "listTime": '["16:50"]'
    }

    return render_template('attendance.html',localStorage_data=localStorage_data)

@app.route('/submit_fill_class',methods=['POST','GET'])
def submit_fill_class():
    class_selected=request.form.getlist('classSelector')
    checked_students = request.form.getlist('students')
    rollList = request.form.getlist('localStorageData')[0].split(',')
    print('---------submit_fill_class starts---------')
    print(rollList)
    print(checked_students)
    print(class_selected)
    for each in checked_students:
        if each not in rollList:
            try:
                print('inside try')
                with engine.connect() as conn:
                    for each in checked_students:
                        if each not in rollList:
                            print(f"Inserting student {each} into class {class_selected[0]}")
                            insert_query = sqlalchemy.text("""
                                INSERT INTO enrollment (studentID, subject_code)
                                VALUES (:studentID, :class_code)
                            """)
                            conn.execute(insert_query, {"studentID": int(each), "class_code": class_selected[0]})

                    conn.commit()
            except Exception as e:
                print("Error adding class:" + str(e))
    print('Data inserted')
    
    temp = []
    for each in rollList:
        if each and each not in checked_students:
            try:
                print('inside try')
                with engine.connect() as conn:
                    delete_query = sqlalchemy.text("""
                        DELETE FROM enrollment
                        WHERE studentID = :student_id AND subject_code = :class_code
                    """)
                    result = conn.execute(delete_query, {
                        "student_id": int(each),
                        "class_code": class_selected[0]
                    })
                    print("deleted")
                    print(result.rowcount)
                    conn.commit()

                temp.append(int(each))

            except Exception as e:
                print("message :" + str(e))    
    data={
        "class_id":class_selected[0],
        "employee_id" : {"in":temp}
    }
    print(json.dumps(data))
    print('---------submit_fill_class ends---------')
    #responsedel = JSON.deleteJSONCALL(f'https://us-east-2.aws.neurelo.com/rest/enrollment_link?filter={json.dumps(data)}','','DELETE')
    return redirect(url_for('attendance'))

def retrive_data(classes):
    students_data = {}

    with engine.connect() as conn:
        for subject_code in classes:
            students_data[subject_code] = []

            # Get enrolled students for that classID
            enrollment_query = sqlalchemy.text("""
                SELECT s.StudentID, s.fname, s.Lname
                FROM enrollment e
                JOIN Student s ON e.studentID = s.StudentID
                WHERE e.subject_code = :subject_code
            """)
            students = conn.execute(enrollment_query, {"subject_code": subject_code}).fetchall()

            for student in students:
                student_info = {
                    "name": f"{student[1]} {student[2]}",
                    "rollNumber": student[0]
                }
                students_data[subject_code].append(student_info)

    return students_data

def list_classes():
    classes = []
    try:
        with engine.connect() as conn:
            query = sqlalchemy.text("SELECT subject_code, start_time FROM class")
            result = conn.execute(query)
            print(result)
            for row in result:
                classes.append(row[0])
                classes.append(row[1])

        return classes

    except Exception as e:
        print(f"Error fetching class data: {e}")
        return []


@app.route('/fill_class')
def fill_class():
    freshData = []

    try:
        with engine.connect() as conn:
            query = sqlalchemy.text("SELECT fname, Lname, StudentID FROM Student")
            result = conn.execute(query)

            for row in result:
                temp = []
                temp.append(row[0])
                temp.append(row[1])
                temp.append(row[2])
                freshData.append(temp)

    except Exception as e:
        print(f"Error fetching student data: {e}")


    classes=list_classes()
    listclass=[]
    listTime=[]
    i=0
    for each in classes:
        if i%2==0:
            listclass.append(each)
        else:
            listTime.append(each)
        i=i+1
    print(listclass)
    print(json.dumps(retrive_data(listclass)));
    localStorage_data = {
        "classes": listclass,
        "colors": '{}',
        "debug": "honey:core-sdk:*",
        "students": json.dumps(retrive_data(listclass)),
        "toAssignstud" : freshData,
        "listTime":listTime
    }
    print(localStorage_data)
    return render_template('fill_class.html',localStorage_data=localStorage_data)

@app.route('/calendar')
def calendar():
    # if 'username' not in session:
    #         error_message = "Please Login"
    #         return render_template('login.html', error_message=error_message)
    return render_template('cal.html')

@app.route('/handle_frameData',methods=['POST'])
def handle_frameData():
    print('start')
    data = request.json
    image_data = data.get('image_data')
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []

    dataset_path = 'Dataset/'
    bucket_name = 'attendance-storage'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT StudentID, photo_path FROM Student WHERE photo_path IS NOT NULL"))
        students = result.fetchall()


    for row in students:
        student_id = row[0]
        photo_path = row[1]  # e.g., gs://attendance-storage/123.jpg

        # Extract filename from path
        filename = os.path.basename(photo_path)
        blob = bucket.blob(filename)

        if not blob.exists():
            print(f"Image not found in bucket for {student_id}: {filename}")
            continue

        # Read blob data into memory
        img_bytes = blob.download_as_bytes()
        img = face_recognition.load_image_file(io.BytesIO(img_bytes))

        encodings = face_recognition.face_encodings(img)
        if len(encodings) == 0:
            print(f"No face found in image for {student_id}")
            continue

        known_face_encodings.append(encodings[0])
        known_face_names.append(str(student_id))

    print('Face encoding complete')

    #retrive Data
    # for filename in os.listdir(dataset_path):
    #     file_path = os.path.join(dataset_path, filename)
    #     img = face_recognition.load_image_file(file_path)
    #     face_encoding = face_recognition.face_encodings(img)[0]
    #     if len(face_encoding) == 0:
    #         print(f"No face found in {filename}. Skipping.")
    #         continue
    #     known_face_encodings.append(face_encoding)
    #     name = os.path.splitext(filename)[0]
    #     known_face_names.append(name)
    
    print('error')
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    # Receive frame data from client
    frame_data = image_data
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
    name = "Unknown"
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        # print(name)
        face_names.append(name)
    return name

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get('PORT',8080)))