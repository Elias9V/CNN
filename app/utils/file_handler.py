import os
import shutil
from fastapi import UploadFile

def save_uploaded_file(file: UploadFile):
    save_path = os.path.join("data/uploads", file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return save_path
