import pickle
import cv2
import os
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db, storage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get values from the .env file
firebase_db_url = os.getenv("FIREBASE_DATABASE_URL")
firebase_storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS")

cred = credentials.Certificate(firebase_credentials_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': firebase_db_url,
    'storageBucket': firebase_storage_bucket
 }
)

# Import student images

folderPath = 'images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
# print(modePathList)

for imgPath in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, imgPath)))
    studentIds.append(os.path.splitext(imgPath)[0])

    fileName = f'{folderPath}/{imgPath}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

    # print(imgPath)
    # print(os.path.splitext(imgPath)[0])
# print(imgList)
print(studentIds)



def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        # img = cv2.resize(img, (216, 216))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (216, 216))
        encode = face_recognition.face_encodings(img)[0]
        # if encodings:
        #     encode = encodings[0]
        #     encodeList.append(encode)
        encodeList.append(encode)

    return encodeList

print("Encoding Started ....")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print(encodeListKnown)
print("Encoding Complete.")


file = open("EncodFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File saved.")

