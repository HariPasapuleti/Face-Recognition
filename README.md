# Face Recognition System

This GitHub repository contains a Face Recognition System built using Python, OpenCV, and Firebase. The system can recognize faces, manage attendance records, and store user data in a Firebase Realtime Database. It includes two main Python scripts: `Face_Recognition.py` and `EncodeGenerator.py`.

---

## Features
- **Real-Time Face Recognition:** Detects and recognizes faces using the `face_recognition` library.
- **Attendance Management:** Logs attendance with timestamps and updates total attendance count.
- **Firebase Integration:** Uses Firebase Realtime Database and Cloud Storage for data storage and retrieval.
- **Encoding Generation:** Generates face encodings from images and stores them in a pickle file for quick access.

---

## Prerequisites
### 1. Python Dependencies
Install the following packages:
```bash
pip install opencv-python face_recognition numpy cvzone firebase-admin python-dotenv
```

### 2. Firebase Setup
- **Realtime Database**: Enable and configure Firebase Realtime Database.
- **Cloud Storage**: Configure Firebase Cloud Storage.
- **Credentials**: Download the Firebase credentials JSON file and place it in the project directory.

### 3. Environment Variables
Create a `.env` file and add:
```env
FIREBASE_DATABASE_URL = "your-firebase-database-url"
FIREBASE_STORAGE_BUCKET = "your-firebase-storage-bucket"
FIREBASE_CREDENTIALS = "path-to-your-firebase-credentials.json"
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HariPasapuleti/Face-Recognition.git
   cd Face-Recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up resources:
   - Create a `resource` folder and add:
     - A background image (`background.png`).
     - A `modes` folder containing mode images.
   - Create an `images` folder for student images.

---

## Usage
### Step 1: Generate Encodings
Run the `EncodeGenerator.py` script to generate face encodings from student images:
```bash
python EncodeGenerator.py
```
This script:
- Uploads student images to Firebase Storage.
- Generates a `EncodFile.p` file containing face encodings and IDs.

### Step 2: Run Face Recognition
Run the `face_recognition.py` script to start the recognition system:
```bash
python Face_Recognition.py
```
Press `n` to exit the program.

---

## File Structure
### Main Files
- **`EncodeGenerator.py`:**
  - Generates face encodings and uploads images to Firebase.
- **`Face_Recognition.py`:**
  - Detects faces in real time, recognizes them, and logs attendance.

### Resources
- **`resource/`:**
  - `background.png`: Background image for UI.
  - `modes/`: Images representing different modes in the UI.
- **`images/`:**
  - Stores student images for encoding.

---

## Firebase Configuration
### Realtime Database Structure
```json
{
  "Students": {
    "123": {
      "name": "Hari Pasapuleti",
      "gender": "Male",
      "company": "XYZ Corp",
      "id": "123",
      "nationality": "India",
      "Joined year": "2023",
      "total attendance": 5,
      "last attendance time": "2025-02-01 10:00:00"
    }
  }
}
```

### Cloud Storage
Store student images in the `images/` folder within Firebase Storage.

---

## Screenshots
Add relevant screenshots showcasing:
- Application UI.
- Firebase setup.
- Sample outputs.

---

## Contributing
Contributions are welcome! Please:
1. Open an issue for bugs or feature requests.
2. Submit a pull request for fixes or enhancements.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- [OpenCV](https://opencv.org/)
- [Face Recognition Library](https://github.com/ageitgey/face_recognition)
- [Firebase](https://firebase.google.com/)

---

## Contact
For any queries, feel free to reach out via [Email](mailto:hari9000kmph@gmail.com).

---

### Optimizations:
1. Improved section organization with clear headings.
2. Simplified repetitive instructions into concise steps.
3. Clarified resource and setup details for better user experience.
4. Added proper Markdown formatting for readability.
