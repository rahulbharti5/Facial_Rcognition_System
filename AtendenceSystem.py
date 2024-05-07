import cv2
import os
import numpy as np
import csv

# Function to detect faces in an image using OpenCV's pre-trained Haar Cascade classifier
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces, gray

# Function to train the face recognition model
def train_faces(data_dir):
    faces = []
    face_ids = []
    names = {}
    current_id = 0

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                if not label in names:
                    names[label] = current_id
                    current_id += 1
                id_ = names[label]
                pil_image = cv2.imread(path)
                gray_image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                image_array = np.array(gray_image, "uint8")
                faces.append(image_array)
                face_ids.append(id_)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(face_ids))
    return recognizer, names

# Function to add new faces to the dataset
def add_face(data_dir, name, image):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cv2.imwrite(os.path.join(data_dir, f"{name}.jpg"), image)

# Function to generate attendance CSV file
def generate_attendance_csv(attendance_dict):
    with open('attendance.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Present'])
        for name, present in attendance_dict.items():
            writer.writerow([name, present])

# Main function
def main():
    data_dir = 'dataset'
    cap = cv2.VideoCapture(0)

    # Train face recognition model
    recognizer, names = train_faces('data_dir')
    attendance_dict = {}

    # Add new faces to dataset (if needed)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Add New Face (Press "c" to capture)', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            name = input("Enter the name of the person: ")
            add_face(data_dir, name, frame)
            print(f"Face of {name} added successfully.")
            break

    # Recognize faces and take attendance
    while True:
        ret, frame = cap.read()
        faces, gray = detect_faces(frame)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi_gray)
            if confidence < 70:
                name = [name for name, label in names.items() if label == id_][0]
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                attendance_dict.setdefault(name, False)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Update attendance
    for name in attendance_dict:
        if input(f"Is {name} present? (y/n): ").lower() == 'y':
            attendance_dict[name] = True

    # Generate attendance CSV file
    generate_attendance_csv(attendance_dict)
    print("Attendance CSV file generated successfully.")

if __name__ == "__main__":
    main()
