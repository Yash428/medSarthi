import bcrypt
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from src.config import settings

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # encode strings and safely truncate to 72 bytes to avoid bcrypt strict ValueErrors
    encoded_plain = plain_password.encode('utf-8')[:72]
    encoded_hashed = hashed_password.encode('utf-8')
    return bcrypt.checkpw(encoded_plain, encoded_hashed)

def get_password_hash(password: str) -> str:
    encoded_password = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(encoded_password, salt).decode('utf-8')

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt
