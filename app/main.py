import os
import io
import csv
import time
import hmac
import hashlib
import base64

from pathlib import Path

from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from app.rag import generate_rag_answer

from app.database import (
    initialize_database,
    create_lead,
    get_leads,
    get_lead_stats,
    get_monthly_lead_counts,
    get_leads_by_date,
)


# ================================================================
# BASE DIRECTORY
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"

ENV_FILE = BASE_DIR / ".env"


# ================================================================
# LOAD ENVIRONMENT VARIABLES
# ================================================================

load_dotenv(ENV_FILE)


# ================================================================
# FASTAPI APP
# ================================================================

app = FastAPI(
    title="Vamadeva AI Assistant",
    description="AI chatbot and enquiry management system for Vamadeva Techno Solutions.",
    version="1.0.0",
)


# ================================================================
# CORS
# ================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# ADMIN CONFIGURATION
# ================================================================

ADMIN_USERNAME = os.getenv(
    "ADMIN_USERNAME",
    "admin",
)

ADMIN_PASSWORD = os.getenv(
    "ADMIN_PASSWORD",
    "",
)

ADMIN_SESSION_SECRET = os.getenv(
    "ADMIN_SESSION_SECRET",
    "change-this-development-secret",
)

ADMIN_COOKIE_NAME = "vamadeva_admin_session"

ADMIN_SESSION_DURATION = 8 * 60 * 60


# ================================================================
# DATABASE INITIALIZATION
# ================================================================

initialize_database()


# ================================================================
# REQUEST MODELS
# ================================================================

class ChatRequest(BaseModel):
    message: str

    course: str | None = None

    current_course: str | None = None

    conversation_history: list = []


class LeadRequest(BaseModel):
    name: str

    phone: str

    email: str | None = None

    qualification: str | None = None

    career_goal: str | None = None

    experience: str | None = None

    course: str | None = None

    training_mode: str | None = None

    message: str | None = None


class AdminLoginRequest(BaseModel):
    username: str

    password: str


# ================================================================
# ADMIN SESSION HELPERS
# ================================================================

def create_admin_session():
    """
    Creates a signed admin session token.

    The token contains:
        timestamp.signature
    """

    timestamp = str(int(time.time()))

    message = (
        ADMIN_USERNAME
        + ":"
        + timestamp
    )

    signature = hmac.new(
        ADMIN_SESSION_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    token = (
        timestamp
        + "."
        + signature
    )

    return base64.urlsafe_b64encode(
        token.encode("utf-8")
    ).decode("utf-8")


def verify_admin_session(token: str | None):
    if not token:
        return False

    try:
        decoded = base64.urlsafe_b64decode(
            token.encode("utf-8")
        ).decode("utf-8")

        parts = decoded.split(".")

        if len(parts) != 2:
            return False

        timestamp = parts[0]

        provided_signature = parts[1]

        timestamp_int = int(timestamp)

        current_time = int(time.time())

        if (
            current_time - timestamp_int
            > ADMIN_SESSION_DURATION
        ):
            return False

        if timestamp_int > current_time:
            return False

        message = (
            ADMIN_USERNAME
            + ":"
            + timestamp
        )

        expected_signature = hmac.new(
            ADMIN_SESSION_SECRET.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(
            provided_signature,
            expected_signature,
        )

    except Exception:
        return False


def is_admin_authenticated(
    request: Request,
):
    token = request.cookies.get(
        ADMIN_COOKIE_NAME
    )

    return verify_admin_session(token)


def require_admin(
    request: Request,
):
    if not is_admin_authenticated(request):

        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "message": "Admin authentication required.",
            },
        )

    return None


# ================================================================
# HOME
# ================================================================

@app.get("/")
def home():

    index_file = (
        FRONTEND_DIR
        /
        "index.html"
    )

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found.",
        )

    return FileResponse(
        index_file
    )


# ================================================================
# VAMADEVA LOGO
# ================================================================

@app.get("/vamadeva-logo.jpg")
def vamadeva_logo_jpg():

    logo_file = (
        FRONTEND_DIR
        /
        "vamadeva-logo.jpg"
    )

    if not logo_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Vamadeva logo JPG not found.",
        )

    return FileResponse(
        logo_file,
        media_type="image/jpeg",
    )


@app.get("/vamadeva-logo.jpeg")
def vamadeva_logo_jpeg():

    logo_file = (
        FRONTEND_DIR
        /
        "vamadeva-logo.jpg"
    )

    if not logo_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Vamadeva logo JPG not found.",
        )

    return FileResponse(
        logo_file,
        media_type="image/jpeg",
    )


# ================================================================
# ADMIN LOGIN LOGO
# ================================================================

@app.get("/admin/vamadeva-logo.jpg")
def admin_vamadeva_logo_jpg():

    logo_file = (
        FRONTEND_DIR
        /
        "vamadeva-logo.jpg"
    )

    if not logo_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Vamadeva logo JPG not found.",
        )

    return FileResponse(
        logo_file,
        media_type="image/jpeg",
    )


@app.get("/admin/vamadeva-logo.jpeg")
def admin_vamadeva_logo_jpeg():

    logo_file = (
        FRONTEND_DIR
        /
        "vamadeva-logo.jpg"
    )

    if not logo_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Vamadeva logo JPG not found.",
        )

    return FileResponse(
        logo_file,
        media_type="image/jpeg",
    )


# ================================================================
# CHAT
# ================================================================

@app.post("/chat")
def chat(
    request: ChatRequest,
):

    question = (
        request.message.strip()
    )

    if not question:

        return {
            "success": False,
            "answer": "Please enter a question.",
        }

    current_course = (
        request.current_course
        or request.course
    )

    conversation_history = (
        request.conversation_history
        or []
    )

    try:

        result = generate_rag_answer(
            question,
            current_course,
            conversation_history,
        )

        # ========================================================
        # IMPORTANT FIX
        # ========================================================
        #
        # generate_rag_answer() returns:
        #
        # {
        #     "answer": "...",
        #     "course": "Python"
        # }
        #
        # Previously the entire dictionary was placed inside
        # "answer", which caused the frontend to display:
        #
        # [object Object]
        #
        # We now return the actual answer string separately.
        # ========================================================

        if isinstance(result, dict):

            answer_text = result.get(
                "answer",
                "",
            )

            detected_course = result.get(
                "course",
                current_course,
            )

        else:

            answer_text = str(result)

            detected_course = current_course

        return {
            "success": True,
            "answer": answer_text,
            "course": detected_course,
        }

    except Exception as error:

        print(
            "[CHAT ERROR]",
            error,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "answer": (
                    "I'm sorry, I couldn't process "
                    "your question right now. "
                    "Please try again."
                ),
            },
        )


# ================================================================
# LEAD SUBMISSION
# ================================================================

@app.post("/leads")
def submit_lead(
    lead: LeadRequest,
):

    name = (
        lead.name.strip()
    )

    phone = (
        lead.phone.strip()
    )

    if not name:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Name is required.",
            },
        )

    if not phone:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Phone number is required.",
            },
        )

    try:

        lead_id = create_lead(

            name=name,

            phone=phone,

            email=lead.email,

            qualification=(
                lead.qualification
            ),

            career_goal=(
                lead.career_goal
            ),

            experience=(
                lead.experience
            ),

            course=(
                lead.course
            ),

            training_mode=(
                lead.training_mode
            ),

            message=(
                lead.message
            ),
        )

        return {
            "success": True,
            "message": (
                "Your enquiry has been "
                "submitted successfully."
            ),
            "lead_id": lead_id,
        }

    except Exception as error:

        print(
            "[LEAD ERROR]",
            error,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Unable to submit your "
                    "enquiry right now."
                ),
            },
        )


# ================================================================
# ADMIN LOGIN PAGE
# ================================================================

@app.get("/admin/login")
def admin_login_page(
    request: Request,
):

    if is_admin_authenticated(
        request
    ):

        return RedirectResponse(
            url="/admin",
            status_code=303,
        )

    login_file = (
        FRONTEND_DIR
        /
        "admin-login.html"
    )

    if not login_file.exists():

        raise HTTPException(
            status_code=404,
            detail="admin-login.html not found.",
        )

    return FileResponse(
        login_file
    )


# ================================================================
# ADMIN LOGIN
# ================================================================

@app.post("/admin/login")
def admin_login(
    login: AdminLoginRequest,
):

    username = (
        login.username.strip()
    )

    password = login.password

    if not ADMIN_PASSWORD:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Admin password is not "
                    "configured in .env."
                ),
            },
        )

    username_correct = hmac.compare_digest(
        username,
        ADMIN_USERNAME,
    )

    password_correct = hmac.compare_digest(
        password,
        ADMIN_PASSWORD,
    )

    if not (
        username_correct
        and password_correct
    ):

        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "message": (
                    "Invalid username or password."
                ),
            },
        )

    token = (
        create_admin_session()
    )

    result = JSONResponse(
        content={
            "success": True,
            "message": "Login successful.",
        }
    )

    result.set_cookie(
        key=ADMIN_COOKIE_NAME,
        value=token,
        max_age=ADMIN_SESSION_DURATION,
        httponly=True,
        samesite="lax",
        secure=False,
    )

    return result


# ================================================================
# ADMIN LOGOUT
# ================================================================

@app.post("/admin/logout")
def admin_logout():

    result = JSONResponse(
        content={
            "success": True,
            "message": "Logged out successfully.",
        }
    )

    result.delete_cookie(
        key=ADMIN_COOKIE_NAME
    )

    return result


# ================================================================
# ADMIN DASHBOARD
# ================================================================

@app.get("/admin")
def admin_dashboard(
    request: Request,
):

    if not is_admin_authenticated(
        request
    ):

        return RedirectResponse(
            url="/admin/login",
            status_code=303,
        )

    admin_file = (
        FRONTEND_DIR
        /
        "admin.html"
    )

    if not admin_file.exists():

        raise HTTPException(
            status_code=404,
            detail="admin.html not found.",
        )

    return FileResponse(
        admin_file
    )


# ================================================================
# ADMIN - ALL LEADS
# ================================================================

@app.get("/admin/leads")
def admin_leads(
    request: Request,
):

    auth_error = require_admin(
        request
    )

    if auth_error:

        return auth_error

    try:

        leads = get_leads()

        return {
            "success": True,
            "count": len(leads),
            "leads": leads,
        }

    except Exception as error:

        print(
            "[ADMIN LEADS ERROR]",
            error,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Unable to load enquiries."
                ),
            },
        )


# ================================================================
# ADMIN - STATISTICS
# ================================================================

@app.get("/admin/stats")
def admin_stats(
    request: Request,
):

    auth_error = require_admin(
        request
    )

    if auth_error:

        return auth_error

    try:

        stats = get_lead_stats()

        return {
            "success": True,

            "total": stats.get(
                "total",
                0,
            ),

            "today": stats.get(
                "today",
                0,
            ),

            "this_month": stats.get(
                "this_month",
                0,
            ),

            "courses": stats.get(
                "courses",
                [],
            ),
        }

    except Exception as error:

        print(
            "[ADMIN STATS ERROR]",
            error,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Unable to load statistics."
                ),
            },
        )


# ================================================================
# ADMIN - MONTHLY CALENDAR
# ================================================================

@app.get("/admin/calendar")
def admin_calendar(
    request: Request,
    year: int,
    month: int,
):

    auth_error = require_admin(
        request
    )

    if auth_error:

        return auth_error

    if month < 1 or month > 12:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": (
                    "Month must be between "
                    "1 and 12."
                ),
            },
        )

    try:

        counts = (
            get_monthly_lead_counts(
                year,
                month,
            )
        )

        return {
            "success": True,
            "year": year,
            "month": month,
            "counts": counts,
        }

    except Exception as error:

        print(
            "[ADMIN CALENDAR ERROR]",
            error,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Unable to load calendar."
                ),
            },
        )


# ================================================================
# ADMIN - LEADS BY DATE
# ================================================================

@app.get("/admin/leads/date")
def admin_leads_by_date(
    request: Request,
    selected_date: str,
):

    auth_error = require_admin(
        request
    )

    if auth_error:

        return auth_error

    if not selected_date:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": (
                    "Selected date is required."
                ),
            },
        )

    try:

        leads = get_leads_by_date(
            selected_date
        )

        return {
            "success": True,
            "date": selected_date,
            "count": len(leads),
            "leads": leads,
        }

    except Exception as error:

        print(
            "[ADMIN DATE LEADS ERROR]",
            error,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Unable to load enquiries "
                    "for the selected date."
                ),
            },
        )


# ================================================================
# CSV HELPER
# ================================================================

def create_csv_response(
    leads,
    filename,
):

    output = io.StringIO()

    writer = csv.writer(
        output
    )

    writer.writerow([
        "ID",
        "Name",
        "Phone",
        "Email",
        "Qualification",
        "Career Goal",
        "Experience",
        "Course",
        "Training Mode",
        "Message",
        "Created At",
    ])

    for lead in leads:

        writer.writerow([

            lead.get(
                "id",
                "",
            ),

            lead.get(
                "name",
                "",
            ),

            lead.get(
                "phone",
                "",
            ),

            lead.get(
                "email",
                "",
            ),

            lead.get(
                "qualification",
                "",
            ),

            lead.get(
                "career_goal",
                "",
            ),

            lead.get(
                "experience",
                "",
            ),

            lead.get(
                "course",
                "",
            ),

            lead.get(
                "training_mode",
                "",
            ),

            lead.get(
                "message",
                "",
            ),

            lead.get(
                "created_at",
                "",
            ),
        ])

    csv_content = (
        output.getvalue()
    )

    return StreamingResponse(

        io.BytesIO(
            csv_content.encode(
                "utf-8-sig"
            )
        ),

        media_type=(
            "text/csv; charset=utf-8"
        ),

        headers={
            "Content-Disposition":
                f'attachment; filename="{filename}"'
        },
    )


# ================================================================
# ADMIN - EXPORT ALL CSV
# ================================================================

@app.get("/admin/export.csv")
def admin_export_csv(
    request: Request,
):

    auth_error = require_admin(
        request
    )

    if auth_error:

        return auth_error

    try:

        leads = get_leads()

        return create_csv_response(
            leads,
            "vamadeva_all_enquiries.csv",
        )

    except Exception as error:

        print(
            "[ADMIN EXPORT ERROR]",
            error,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Unable to export enquiries."
                ),
            },
        )


# ================================================================
# ADMIN - EXPORT SELECTED DATE CSV
# ================================================================

@app.get("/admin/export-date.csv")
def admin_export_date_csv(
    request: Request,
    selected_date: str,
):

    auth_error = require_admin(
        request
    )

    if auth_error:

        return auth_error

    if not selected_date:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": (
                    "Selected date is required."
                ),
            },
        )

    try:

        leads = get_leads_by_date(
            selected_date
        )

        safe_date = (
            selected_date.replace(
                "-",
                "",
            )
        )

        filename = (
            "vamadeva_enquiries_"
            f"{safe_date}.csv"
        )

        return create_csv_response(
            leads,
            filename,
        )

    except Exception as error:

        print(
            "[ADMIN DATE EXPORT ERROR]",
            error,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Unable to export "
                    "selected-date enquiries."
                ),
            },
        )


# ================================================================
# HEALTH CHECK
# ================================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
    }


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )