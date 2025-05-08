# app/config.py
import os
from dotenv import load_dotenv, find_dotenv

# Cargar .env
env_path = find_dotenv()
load_dotenv(env_path, override=True)

BASE_URL = os.getenv("BASE_URL")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
