import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer

_ALGORITHM = "HS256"
_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def _secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET environment variable is not set.")
    return secret


def create_access_token(subject: str) -> str:
    """Create a signed JWT with an expiry.

    Args:
        subject: Identity claim (e.g. username).

    Returns:
        Encoded JWT string.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, _secret(), algorithm=_ALGORITHM)


def decode_token(token: str) -> str:
    """Decode and validate a JWT, returning the subject.

    Args:
        token: JWT string from the Authorization header.

    Returns:
        Subject claim from the token.

    Raises:
        HTTPException 401: if token is expired or invalid.
    """
    try:
        payload = jwt.decode(token, _secret(), algorithms=[_ALGORITHM])
        subject: str = payload["sub"]
        return subject
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )
