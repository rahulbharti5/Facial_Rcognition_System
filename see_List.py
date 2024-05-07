import pickle
import numpy as np

# Load the faces data from the pickle file
with open('Data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)

# Load the names associated with the faces
with open('Data/names.pkl', 'rb') as f:
    names = pickle.load(f)

# Print the names and faces data
for name, face_data in zip(names, faces_data):
    print("Name:", name)
    print("Face data:", face_data)
