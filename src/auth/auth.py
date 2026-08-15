from datetime import datetime, timedelta, timezone
from src.auth.database import init_db
import os

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

from src.auth.database import get_connection


# ============================================================
# JWT Configuration
# ============================================================

JWT_SECRET = os.getenv(
    "JWT_SECRET",
    "CHANGE_THIS_TO_A_LONG_RANDOM_SECRET"
)

JWT_ALGORITHM = "HS256"

# Token valid for 24 hours
JWT_EXPIRE_MINUTES = 60 * 24


# ============================================================
# Password Hashing
# ============================================================

password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """
    Hash a user's password using Argon2.
    """
    return password_hasher.hash(password)


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    """
    Verify a password against its Argon2 hash.
    """

    try:
        return password_hasher.verify(
            password_hash,
            password,
        )

    except (VerifyMismatchError, InvalidHashError):
        return False


def hash_recovery_code(code: str) -> str:
    """
    Hash recovery code using Argon2.
    """

    return password_hasher.hash(code)


# ============================================================
# JWT Token
# ============================================================

def create_token(
    user_id: int,
    username: str,
) -> str:
    """
    Create JWT authentication token.
    """

    expires_at = (
        datetime.now(timezone.utc)
        + timedelta(minutes=JWT_EXPIRE_MINUTES)
    )

    payload = {
        "user_id": int(user_id),
        "username": username,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def decode_token(token: str):
    """
    Decode and validate JWT token.

    Returns:
        payload dictionary if valid
        None if invalid/expired
    """

    try:

        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )

        return payload

    except jwt.InvalidTokenError:
        return None


# ============================================================
# Create User
# ============================================================

def create_user(
    username: str,
    password: str,
    recovery_code: str,
):
    """
    Create a new user account.
    """

    username = username.strip()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not username:
        raise ValueError(
            "Username is required."
        )

    if len(username) < 3:
        raise ValueError(
            "Username must contain at least 3 characters."
        )

    if len(username) > 50:
        raise ValueError(
            "Username must not exceed 50 characters."
        )

    if not password:
        raise ValueError(
            "Password is required."
        )

    if len(password) < 6:
        raise ValueError(
            "Password must contain at least 6 characters."
        )

    if len(recovery_code) < 6:
        raise ValueError(
            "Recovery code must contain at least 6 characters."
        )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    connection = get_connection()

    try:

        password_hash = hash_password(password)

        recovery_code_hash = hash_recovery_code(
            recovery_code
        )

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
                password_hash,
                recovery_code_hash,
                datetime.now(
                    timezone.utc
                ).isoformat(),
            ),
        )

        connection.commit()

        user_id = cursor.lastrowid

        return {
            "id": int(user_id),
            "username": username,
        }

    except Exception as error:

        connection.rollback()

        if "UNIQUE constraint failed" in str(error):

            raise ValueError(
                "Username already exists."
            )

        raise

    finally:

        connection.close()


# ============================================================
# Authenticate User
# ============================================================

def authenticate_user(
    username: str,
    password: str,
):
    """
    Authenticate username/password.

    Returns:
        user dictionary if valid
        None if invalid
    """

    username = username.strip()

    connection = get_connection()

    try:

        user = connection.execute(
            """
            SELECT
                id,
                username,
                password_hash
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

        if user is None:
            return None

        valid_password = verify_password(
            password,
            user["password_hash"],
        )

        if not valid_password:
            return None

        return {
            "id": int(user["id"]),
            "username": user["username"],
        }

    finally:

        connection.close()


# ============================================================
# Verify Recovery Code
# ============================================================

def verify_recovery_code(
    username: str,
    recovery_code: str,
) -> bool:
    """
    Verify recovery code for a user.
    """

    username = username.strip()

    connection = get_connection()

    try:

        user = connection.execute(
            """
            SELECT recovery_code_hash
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

        if user is None:
            return False

        try:

            return password_hasher.verify(
                user["recovery_code_hash"],
                recovery_code,
            )

        except (
            VerifyMismatchError,
            InvalidHashError,
        ):

            return False

    finally:

        connection.close()


# ============================================================
# Reset Password
# ============================================================

def reset_password(
    username: str,
    new_password: str,
) -> bool:
    """
    Reset user's password.

    Recovery-code verification should be performed
    before calling this function.
    """

    username = username.strip()

    if not username:
        return False

    if len(new_password) < 6:
        raise ValueError(
            "Password must contain at least 6 characters."
        )

    connection = get_connection()

    try:

        password_hash = hash_password(
            new_password
        )

        cursor = connection.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE username = ?
            """,
            (
                password_hash,
                username,
            ),
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:

        connection.close()