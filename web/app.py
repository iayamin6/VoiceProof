"""VoiceProof web application: Google sign-in, private uploads, inference history."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .inference import analyze_audio, load_model

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "voiceproof.db"))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
oauth = OAuth()
SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    if os.getenv("DEV_MODE"):
        SESSION_SECRET = "development-only-secret-change-before-deploying"
    else:
        raise RuntimeError("Set SESSION_SECRET. Refusing to start with an insecure session key.")
if os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"):
    oauth.register(
        name="google",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialise_database() -> None:
    with db() as connection:
        connection.executescript("""
        CREATE TABLE IF NOT EXISTS users (
          google_sub TEXT PRIMARY KEY, email TEXT NOT NULL, name TEXT, picture TEXT,
          created_at TEXT NOT NULL, last_login_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analyses (
          id INTEGER PRIMARY KEY AUTOINCREMENT, google_sub TEXT NOT NULL,
          original_filename TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL,
          FOREIGN KEY (google_sub) REFERENCES users(google_sub)
        );
        """)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialise_database()
    try:
        app.state.model = load_model()
        app.state.model_error = None
    except RuntimeError as exc:
        app.state.model = None
        app.state.model_error = str(exc)
    yield


app = FastAPI(title="VoiceProof", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, https_only=not os.getenv("DEV_MODE"))
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def current_user(request: Request) -> dict | None:
    return request.session.get("user")


def require_user(request: Request) -> dict:
    user = current_user(request)
    if not user:
        raise HTTPException(401, "Sign in with Google to analyse audio.")
    return user


@app.get("/")
async def home(request: Request):
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/session")
async def session(request: Request):
    """Lets the static UI adapt without server-side template syntax."""
    return {"user": current_user(request), "model_ready": app.state.model is not None, "model_error": app.state.model_error}


@app.get("/login")
async def login(request: Request):
    if not os.getenv("GOOGLE_CLIENT_ID") or not os.getenv("GOOGLE_CLIENT_SECRET"):
        raise HTTPException(503, "Google OAuth is not configured. Add the Google client credentials to .env.")
    return await oauth.google.authorize_redirect(request, request.url_for("auth_callback"))


@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    profile = token.get("userinfo")
    if not profile or not profile.get("sub") or not profile.get("email"):
        raise HTTPException(400, "Google did not return a valid user profile.")
    now = datetime.now(UTC).isoformat()
    user = {"sub": profile["sub"], "email": profile["email"], "name": profile.get("name", profile["email"]), "picture": profile.get("picture")}
    with db() as connection:
        connection.execute("""INSERT INTO users (google_sub,email,name,picture,created_at,last_login_at)
          VALUES (:sub,:email,:name,:picture,:now,:now)
          ON CONFLICT(google_sub) DO UPDATE SET email=:email,name=:name,picture=:picture,last_login_at=:now""", {**user, "now": now})
    request.session["user"] = user
    return RedirectResponse("/", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


@app.get("/api/history")
async def history(request: Request):
    user = require_user(request)
    with db() as connection:
        records = connection.execute("SELECT id, original_filename, result_json, created_at FROM analyses WHERE google_sub=? ORDER BY id DESC LIMIT 20", (user["sub"],)).fetchall()
    return [{"id": r["id"], "filename": r["original_filename"], "created_at": r["created_at"], **json.loads(r["result_json"])} for r in records]


@app.post("/api/analyze")
async def analyse(request: Request, audio: UploadFile = File(...)):
    user = require_user(request)
    if app.state.model is None:
        raise HTTPException(503, app.state.model_error or "The detection model is unavailable.")
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Use WAV, MP3, M4A, FLAC, or OGG audio.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temp_path = Path(temporary.name)
        copied = 0
        while chunk := await audio.read(1024 * 1024):
            copied += len(chunk)
            if copied > MAX_UPLOAD_BYTES:
                temporary.close(); temp_path.unlink(missing_ok=True)
                raise HTTPException(413, "Audio must be 25 MB or smaller.")
            temporary.write(chunk)
    try:
        result = analyze_audio(temp_path, app.state.model)
    except Exception as exc:
        raise HTTPException(422, f"We could not decode or analyse this audio: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)  # Audio is never persisted in the database.

    now = datetime.now(UTC).isoformat()
    safe_name = Path(audio.filename or "audio").name
    with db() as connection:
        cursor = connection.execute("INSERT INTO analyses (google_sub, original_filename, result_json, created_at) VALUES (?,?,?,?)", (user["sub"], safe_name, json.dumps(result), now))
    return JSONResponse({"id": cursor.lastrowid, "filename": safe_name, "created_at": now, **result})
