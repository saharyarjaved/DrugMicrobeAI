from datetime import datetime, timedelta, timezone
import os

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from src.auth.database import get_connection


JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_THIS_TO_A_LONG_RANDOM_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24

password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return password_hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def hash_recovery_code(code: str) -> str:
    return password_hasher.hash(code)


def create_token(user_id: int, username: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=JWT_EXPIRE_MINUTES
    )

    payload = {
        "user_id": user_id,
        "username": username,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def create_user(
    username: str,
    password: str,
    recovery_code: str,
):
    username = username.strip()

    if not username:
        raise ValueError("Username is required.")

    if len(username) < 3:
        raise ValueError(
            "Username must contain at least 3 characters."
        )

    if len(password) < 6:
        raise ValueError(
            "Password must contain at least 6 characters."
        )

    if len(recovery_code) < 6:
        raise ValueError(
            "Recovery code must contain at least 6 characters."
        )

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO users (
                username,
                password_hash,
                recovery_code_hash,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                hash_password(password),
                hash_recovery_code(recovery_code),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        connection.commit()

        user_id = cursor.lastrowid

        return {
            "id": user_id,
            "username": username,
        }

    except Exception as error:
        connection.rollback()

        if "UNIQUE constraint failed" in str(error):
            raise ValueError("Username already exists.")

        raise

    finally:
        connection.close()


def authenticate_user(
    username: str,
    password: str,
):
    connection = get_connection()

    try:
        user = connection.execute(
            """
            SELECT id, username, password_hash
            FROM users
            WHERE username = ?
            """,
            (username.strip(),),
        ).fetchone()

        if user is None:
            return None

        if not verify_password(
            password,
            user["password_hash"],
        ):
            return None

        return {
            "id": user["id"],
            "username": user["username"],
        }

    finally:
        connection.close()


def verify_recovery_code(
    username: str,
    recovery_code: str,
) -> bool:
    connection = get_connection()

    try:
        user = connection.execute(
            """
            SELECT recovery_code_hash
            FROM users
            WHERE username = ?
            """,
            (username.strip(),),
        ).fetchone()

        if user is None:
            return False

        try:
            return password_hasher.verify(
                user["recovery_code_hash"],
                recovery_code,
            )
        except VerifyMismatchError:
            return False

    finally:
        connection.close()

def decode_token(token: str):
    import jwt

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
        return payload
    except jwt.InvalidTokenError:
        return None

def reset_password(username: str, new_password: str) -> bool:
    connection = get_connection()

    try:
        password_hash = hash_password(new_password)

        cursor = connection.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE username = ?
            """,
            (password_hash, username.strip()),
        )

        connection.commit()
        return cursor.rowcount > 0

    finally:
        connection.close()

