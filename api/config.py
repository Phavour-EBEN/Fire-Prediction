import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Firebase configuration
FIREBASE_CONFIG = {
    "apiKey": os.getenv('FIREBASE_API_KEY'),
    "authDomain": os.getenv('FIREBASE_AUTH_DOMAIN'),
    "databaseURL": os.getenv('FIREBASE_DATABASE_URL'),
    "storageBucket": os.getenv('FIREBASE_STORAGE_BUCKET')
}

# Other configuration variables
DEVICE_UUID = os.getenv('DEVICE_UUID', 'Device uuid:iqD78eGmo7LompLJHfZwm2')
