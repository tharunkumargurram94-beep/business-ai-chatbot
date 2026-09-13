# ================================================================
# VAMADEVA AI CHATBOT - DATABASE
# ================================================================
#
# Persistent lead storage using Supabase REST API.
#
# IMPORTANT:
# - SUPABASE_URL must be configured in .env / Render
# - SUPABASE_SERVICE_ROLE_KEY must be configured in .env / Render
# - The service-role key is SERVER-SIDE ONLY.
# - Never expose the service-role key in frontend JavaScript.
#
# This file keeps backward-compatible function names and return
# structures so the existing main.py and Admin Dashboard continue
# to work without UI changes.
# ================================================================

import os
from datetime import datetime
from pathlib import Path
from collections import Counter

import requests
from dotenv import load_dotenv


# ================================================================
# ENVIRONMENT
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    ""
).strip().rstrip("/")


SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    ""
).strip()


TABLE_NAME = "leads"


# ================================================================
# SUPABASE CONFIGURATION CHECK
# ================================================================

def _check_supabase_config():
    """
    Make sure the Supabase connection settings exist.
    """

    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is not configured."
        )

    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY is not configured."
        )


# ================================================================
# SUPABASE HEADERS
# ================================================================

def _headers():
    """
    Headers used for server-side Supabase REST requests.
    """

    _check_supabase_config()

    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": (
            f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"
        ),
        "Content-Type": "application/json",
    }


# ================================================================
# SUPABASE TABLE URL
# ================================================================

def _table_url():
    """
    Return the REST endpoint for the leads table.
    """

    _check_supabase_config()

    return (
        f"{SUPABASE_URL}"
        f"/rest/v1/{TABLE_NAME}"
    )


# ================================================================
# GENERIC SUPABASE REQUEST
# ================================================================

def _request(
    method,
    url=None,
    params=None,
    json_data=None,
    timeout=30,
):
    """
    Perform a Supabase REST request with useful error messages.
    """

    if url is None:
        url = _table_url()

    response = requests.request(
        method=method,
        url=url,
        headers=_headers(),
        params=params,
        json=json_data,
        timeout=timeout,
    )

    if not response.ok:
        try:
            error_body = response.json()
        except Exception:
            error_body = response.text

        raise RuntimeError(
            f"Supabase request failed "
            f"({response.status_code}): "
            f"{error_body}"
        )

    if not response.text.strip():
        return None

    try:
        return response.json()
    except Exception:
        return response.text


# ================================================================
# INITIALIZE DATABASE
# ================================================================

def initialize_database():
    """
    Supabase tables are created from the Supabase SQL Editor.

    This function intentionally does not attempt to create tables
    automatically because database schema creation should remain
    controlled from Supabase.
    """

    _check_supabase_config()

    return True


# ================================================================
# NORMALIZE LEAD
# ================================================================

def _normalize_lead(lead):
    """
    Keep the lead structure compatible with the existing
    application and Admin Dashboard.
    """

    if not isinstance(lead, dict):
        return {}

    return {
        "id": lead.get("id"),
        "name": lead.get("name") or "",
        "phone": lead.get("phone") or "",
        "email": lead.get("email"),
        "qualification": (
            lead.get("qualification")
            or ""
        ),
        "career_goal": (
            lead.get("career_goal")
            or ""
        ),
        "experience": (
            lead.get("experience")
            or ""
        ),
        "course": (
            lead.get("course")
            or ""
        ),
        "training_mode": (
            lead.get("training_mode")
            or ""
        ),
        "message": (
            lead.get("message")
            or ""
        ),
        "created_at": (
            lead.get("created_at")
            or ""
        ),
    }


# ================================================================
# GET ALL LEADS
# ================================================================

def get_leads():
    """
    Return all leads ordered newest first.
    """

    data = _request(
        "GET",
        params={
            "select": "*",
            "order": "id.desc",
        },
    )

    if not data:
        return []

    return [
        _normalize_lead(lead)
        for lead in data
    ]


# ================================================================
# GET SINGLE LEAD
# ================================================================

def get_lead(lead_id):
    """
    Return one lead by ID.
    """

    data = _request(
        "GET",
        params={
            "select": "*",
            "id": f"eq.{lead_id}",
            "limit": "1",
        },
    )

    if not data:
        return None

    return _normalize_lead(
        data[0]
    )


# ================================================================
# CREATE LEAD
# ================================================================

def create_lead(
    name,
    phone,
    email=None,
    qualification=None,
    career_goal=None,
    experience=None,
    course=None,
    training_mode=None,
    message=None,
):
    """
    Create a new enquiry/lead in Supabase.

    Returns:
        int/str: newly created lead ID
    """

    created_at = datetime.now().strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    payload = {
        "name": name,
        "phone": phone,
        "email": email,
        "qualification": qualification,
        "career_goal": career_goal,
        "experience": experience,
        "course": course,
        "training_mode": training_mode,
        "message": message,
        "created_at": created_at,
    }

    headers = _headers()

    headers["Prefer"] = (
        "return=representation"
    )

    response = requests.post(
        _table_url(),
        headers=headers,
        json=payload,
        timeout=30,
    )

    if not response.ok:
        try:
            error_body = response.json()
        except Exception:
            error_body = response.text

        raise RuntimeError(
            f"Unable to create lead "
            f"({response.status_code}): "
            f"{error_body}"
        )

    result = response.json()

    if not result:
        raise RuntimeError(
            "Supabase created the lead but "
            "did not return the lead ID."
        )

    created_lead = result[0]

    return created_lead.get(
        "id"
    )


# ================================================================
# GET LEAD STATISTICS
# ================================================================

def get_lead_stats():
    """
    Return statistics in the exact structure expected by
    the existing main.py.

    Expected structure:

    {
        "total": 4,
        "today": 1,
        "this_month": 4,
        "courses": [
            {
                "course": "Snowflake",
                "count": 2
            }
        ]
    }
    """

    leads = get_leads()

    now = datetime.now()

    today_string = now.strftime(
        "%Y-%m-%d"
    )

    month_string = now.strftime(
        "%Y-%m"
    )

    today_count = 0
    month_count = 0

    course_counter = Counter()

    for lead in leads:

        created_at = str(
            lead.get(
                "created_at",
                ""
            )
        )

        if created_at.startswith(
            today_string
        ):
            today_count += 1

        if created_at.startswith(
            month_string
        ):
            month_count += 1

        course = str(
            lead.get(
                "course",
                ""
            )
        ).strip()

        if course:
            course_counter[
                course
            ] += 1

    courses = []

    for course, count in course_counter.most_common():

        courses.append(
            {
                "course": course,
                "count": count,
            }
        )

    return {
        "total": len(leads),
        "today": today_count,
        "this_month": month_count,
        "courses": courses,
    }


# ================================================================
# GET MONTHLY LEAD COUNTS
# ================================================================

def get_monthly_lead_counts(
    year,
    month,
):
    """
    Return daily lead counts for a specific month.

    This is used by the Admin Dashboard calendar.

    Example:

    {
        "2026-09-01": 0,
        "2026-09-02": 2,
        "2026-09-03": 1
    }
    """

    leads = get_leads()

    prefix = (
        f"{int(year):04d}-"
        f"{int(month):02d}-"
    )

    counts = {}

    for lead in leads:

        created_at = str(
            lead.get(
                "created_at",
                ""
            )
        )

        if not created_at.startswith(
            prefix
        ):
            continue

        date_part = (
            created_at[:10]
        )

        counts[date_part] = (
            counts.get(
                date_part,
                0
            ) + 1
        )

    return counts


# ================================================================
# GET LEADS BY DATE
# ================================================================

def get_leads_by_date(
    selected_date
):
    """
    Return all leads for a specific date.

    selected_date format:

        YYYY-MM-DD

    Example:

        2026-09-12
    """

    selected_date = str(
        selected_date
    ).strip()

    if not selected_date:
        return []

    leads = get_leads()

    matching_leads = []

    for lead in leads:

        created_at = str(
            lead.get(
                "created_at",
                ""
            )
        )

        if created_at.startswith(
            selected_date
        ):
            matching_leads.append(
                lead
            )

    return matching_leads


# ================================================================
# GET LEADS BY COURSE
# ================================================================

def get_leads_by_course(
    course
):
    """
    Return leads matching a course.
    """

    course = str(
        course
    ).strip().lower()

    if not course:
        return []

    leads = get_leads()

    return [
        lead
        for lead in leads
        if str(
            lead.get(
                "course",
                ""
            )
        ).strip().lower()
        == course
    ]


# ================================================================
# GET LEADS BY TRAINING MODE
# ================================================================

def get_leads_by_training_mode(
    training_mode
):
    """
    Return leads matching a training mode.
    """

    training_mode = str(
        training_mode
    ).strip().lower()

    if not training_mode:
        return []

    leads = get_leads()

    return [
        lead
        for lead in leads
        if str(
            lead.get(
                "training_mode",
                ""
            )
        ).strip().lower()
        == training_mode
    ]


# ================================================================
# GET COURSE COUNTS
# ================================================================

def get_course_counts():
    """
    Return course counts as a dictionary.

    Example:

    {
        "Snowflake": 2,
        "Power BI": 1
    }
    """

    leads = get_leads()

    counter = Counter()

    for lead in leads:

        course = str(
            lead.get(
                "course",
                ""
            )
        ).strip()

        if course:
            counter[course] += 1

    return dict(counter)


# ================================================================
# GET TRAINING MODE COUNTS
# ================================================================

def get_training_mode_counts():
    """
    Return training mode counts as a dictionary.
    """

    leads = get_leads()

    counter = Counter()

    for lead in leads:

        mode = str(
            lead.get(
                "training_mode",
                ""
            )
        ).strip()

        if mode:
            counter[mode] += 1

    return dict(counter)


# ================================================================
# DATABASE HEALTH CHECK
# ================================================================

def database_health_check():
    """
    Verify that the application can communicate with Supabase.
    """

    try:

        data = _request(
            "GET",
            params={
                "select": "id",
                "limit": "1",
            },
        )

        return {
            "success": True,
            "connected": True,
            "records_available": (
                len(data)
                if isinstance(
                    data,
                    list
                )
                else 0
            ),
        }

    except Exception as error:

        return {
            "success": False,
            "connected": False,
            "error": str(error),
        }


# ================================================================
# BACKWARD-COMPATIBILITY ALIASES
# ================================================================

def get_all_leads():
    """
    Backward-compatible alias.
    """

    return get_leads()


def get_stats():
    """
    Backward-compatible alias.
    """

    return get_lead_stats()


# ================================================================
# LOCAL TEST
# ================================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "VAMADEVA DATABASE TEST"
    )

    print(
        "========================================\n"
    )

    try:

        health = database_health_check()

        print(
            "Supabase connection:"
        )

        print(
            health
        )

        print()

        leads = get_leads()

        print(
            "Total leads:",
            len(leads)
        )

        print()

        stats = get_lead_stats()

        print(
            "Statistics:"
        )

        print(
            stats
        )

        print(
            "\n========================================\n"
        )

    except Exception as error:

        print(
            "DATABASE ERROR:"
        )

        print(
            error
        )