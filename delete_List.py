import os

# Remove the pickle files containing faces data and names
os.remove('Data/faces_data.pkl')
os.remove('Data/names.pkl')

print("All faces data and names have been deleted.")
