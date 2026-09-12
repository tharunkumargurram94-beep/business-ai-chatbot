import sqlite3
import re
from pathlib import Path
from datetime import datetime


# ================================================================
# PROJECT PATH
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "leads.db"


# ================================================================
# DATABASE CONNECTION
# ================================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ================================================================
# INITIALIZE DATABASE
# ================================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    # ------------------------------------------------------------
    # CREATE TABLE
    # ------------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            phone TEXT NOT NULL,

            email TEXT,

            qualification TEXT,

            career_goal TEXT,

            experience TEXT,

            course TEXT,

            training_mode TEXT,

            message TEXT,

            created_at TEXT NOT NULL

        )
    """)


    # ------------------------------------------------------------
    # CHECK EXISTING COLUMNS
    # ------------------------------------------------------------

    cursor.execute(
        "PRAGMA table_info(leads)"
    )


    existing_columns = {
        row["name"]
        for row in cursor.fetchall()
    }


    required_columns = {

        "qualification":
            "TEXT",

        "career_goal":
            "TEXT",

        "experience":
            "TEXT",

        "course":
            "TEXT",

        "training_mode":
            "TEXT",

        "message":
            "TEXT",

        "created_at":
            "TEXT",

    }


    # ------------------------------------------------------------
    # ADD MISSING COLUMNS
    # ------------------------------------------------------------

    for column, column_type in required_columns.items():

        if column not in existing_columns:

            cursor.execute(
                f"""
                ALTER TABLE leads
                ADD COLUMN {column}
                {column_type}
                """
            )


    connection.commit()


    # ------------------------------------------------------------
    # MIGRATE OLD COMBINED MESSAGES
    # ------------------------------------------------------------

    migrate_combined_messages(
        connection
    )


    connection.close()


# ================================================================
# PARSE OLD COMBINED MESSAGE
# ================================================================

def parse_combined_message(
    message
):

    """
    Converts an old combined message such as:

        Qualification:
        btech

        Career Goal:
        getting into IT

        Additional Message:
        both

    into:

        qualification = btech
        career_goal   = getting into IT
        message       = both
    """


    if not message:

        return {
            "qualification": None,
            "career_goal": None,
            "message": message,
        }


    text = str(
        message
    ).strip()


    # ------------------------------------------------------------
    # Check whether this is actually a combined message
    # ------------------------------------------------------------

    has_qualification = (
        re.search(
            r"Qualification\s*:",
            text,
            re.IGNORECASE
        )
        is not None
    )


    has_career_goal = (
        re.search(
            r"Career\s*Goal\s*:",
            text,
            re.IGNORECASE
        )
        is not None
    )


    has_additional_message = (
        re.search(
            r"Additional\s*Message\s*:",
            text,
            re.IGNORECASE
        )
        is not None
    )


    if not (
        has_qualification
        or
        has_career_goal
        or
        has_additional_message
    ):

        return {
            "qualification": None,
            "career_goal": None,
            "message": message,
        }


    # ------------------------------------------------------------
    # Extract qualification
    # ------------------------------------------------------------

    qualification_match = re.search(
        r"Qualification\s*:\s*(.*?)(?=\n\s*Career\s*Goal\s*:|\n\s*Additional\s*Message\s*:|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )


    qualification = None


    if qualification_match:

        qualification = (
            qualification_match
            .group(1)
            .strip()
        )


    # ------------------------------------------------------------
    # Extract career goal
    # ------------------------------------------------------------

    career_goal_match = re.search(
        r"Career\s*Goal\s*:\s*(.*?)(?=\n\s*Additional\s*Message\s*:|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )


    career_goal = None


    if career_goal_match:

        career_goal = (
            career_goal_match
            .group(1)
            .strip()
        )


    # ------------------------------------------------------------
    # Extract additional message
    # ------------------------------------------------------------

    additional_message_match = re.search(
        r"Additional\s*Message\s*:\s*(.*)$",
        text,
        re.IGNORECASE | re.DOTALL
    )


    additional_message = None


    if additional_message_match:

        additional_message = (
            additional_message_match
            .group(1)
            .strip()
        )


    # ------------------------------------------------------------
    # Clean empty values
    # ------------------------------------------------------------

    if qualification == "":
        qualification = None


    if career_goal == "":
        career_goal = None


    if additional_message == "":
        additional_message = None


    return {

        "qualification":
            qualification,

        "career_goal":
            career_goal,

        "message":
            additional_message,

    }


# ================================================================
# MIGRATE EXISTING COMBINED MESSAGES
# ================================================================

def migrate_combined_messages(
    connection
):

    cursor = connection.cursor()


    cursor.execute("""
        SELECT
            id,
            qualification,
            career_goal,
            message
        FROM leads
        WHERE message IS NOT NULL
          AND (
              message LIKE '%Qualification:%'
              OR
              message LIKE '%Career Goal:%'
              OR
              message LIKE '%Additional Message:%'
          )
    """)


    rows = cursor.fetchall()


    migrated_count = 0


    for row in rows:

        parsed = parse_combined_message(
            row["message"]
        )


        new_qualification = (
            row["qualification"]
            or
            parsed["qualification"]
        )


        new_career_goal = (
            row["career_goal"]
            or
            parsed["career_goal"]
        )


        # If the message contains the structured
        # fields, replace the combined message
        # with only the additional message.

        new_message = (
            parsed["message"]
        )


        # --------------------------------------------------------
        # Only update when something actually changed
        # --------------------------------------------------------

        if (
            new_qualification
            != row["qualification"]
            or
            new_career_goal
            != row["career_goal"]
            or
            new_message
            != row["message"]
        ):

            cursor.execute(
                """
                UPDATE leads
                SET
                    qualification = ?,
                    career_goal = ?,
                    message = ?
                WHERE id = ?
                """,
                (
                    new_qualification,
                    new_career_goal,
                    new_message,
                    row["id"],
                )
            )


            migrated_count += 1


    connection.commit()


    if migrated_count > 0:

        print(
            f"[DATABASE] Migrated "
            f"{migrated_count} combined lead(s)."
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

    # ------------------------------------------------------------
    # If the frontend sends the old combined message format,
    # automatically separate the fields.
    # ------------------------------------------------------------

    parsed = parse_combined_message(
        message
    )


    if not qualification:

        qualification = (
            parsed["qualification"]
            or
            qualification
        )


    if not career_goal:

        career_goal = (
            parsed["career_goal"]
            or
            career_goal
        )


    # If the message was combined,
    # store only the additional message.

    if (
        parsed["qualification"]
        is not None
        or
        parsed["career_goal"]
        is not None
        or
        re.search(
            r"Additional\s*Message\s*:",
            str(message or ""),
            re.IGNORECASE
        )
    ):

        message = (
            parsed["message"]
        )


    connection = get_connection()

    cursor = connection.cursor()


    created_at = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )


    cursor.execute(
        """
        INSERT INTO leads (

            name,

            phone,

            email,

            qualification,

            career_goal,

            experience,

            course,

            training_mode,

            message,

            created_at

        )

        VALUES (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        )
        """,
        (

            name,

            phone,

            email,

            qualification,

            career_goal,

            experience,

            course,

            training_mode,

            message,

            created_at,

        )
    )


    lead_id = (
        cursor.lastrowid
    )


    connection.commit()

    connection.close()


    return lead_id


# ================================================================
# GET ALL LEADS
# ================================================================

def get_leads():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT

            id,

            name,

            phone,

            email,

            qualification,

            career_goal,

            experience,

            course,

            training_mode,

            message,

            created_at

        FROM leads

        ORDER BY id DESC
        """
    )


    rows = cursor.fetchall()


    connection.close()


    return [
        dict(row)
        for row in rows
    ]


# ================================================================
# GET LEAD STATISTICS
# ================================================================

def get_lead_stats():

    connection = get_connection()

    cursor = connection.cursor()


    # ------------------------------------------------------------
    # TOTAL
    # ------------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM leads
        """
    )


    total = (
        cursor.fetchone()["total"]
    )


    # ------------------------------------------------------------
    # TODAY
    # ------------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS today
        FROM leads
        WHERE date(created_at)
            =
            date('now', 'localtime')
        """
    )


    today = (
        cursor.fetchone()["today"]
    )


    # ------------------------------------------------------------
    # THIS MONTH
    # ------------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS this_month
        FROM leads
        WHERE strftime(
            '%Y-%m',
            created_at
        )
        =
        strftime(
            '%Y-%m',
            'now',
            'localtime'
        )
        """
    )


    this_month = (
        cursor.fetchone()["this_month"]
    )


    # ------------------------------------------------------------
    # COURSE COUNTS
    # ------------------------------------------------------------

    cursor.execute(
        """
        SELECT

            course,

            COUNT(*) AS count

        FROM leads

        WHERE course IS NOT NULL

          AND TRIM(course) != ''

        GROUP BY course

        ORDER BY count DESC
        """
    )


    course_rows = (
        cursor.fetchall()
    )


    connection.close()


    return {

        "total":
            total,

        "today":
            today,

        "this_month":
            this_month,

        "courses": [

            {

                "course":
                    row["course"],

                "count":
                    row["count"],

            }

            for row in course_rows

        ],

    }


# ================================================================
# GET MONTHLY LEAD COUNTS
# ================================================================

def get_monthly_lead_counts(
    year,
    month
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT

            CAST(
                strftime(
                    '%d',
                    created_at
                )
                AS INTEGER
            ) AS day,

            COUNT(*) AS count

        FROM leads

        WHERE
            strftime(
                '%Y',
                created_at
            )
            = ?

          AND
            strftime(
                '%m',
                created_at
            )
            = ?

        GROUP BY
            strftime(
                '%d',
                created_at
            )

        ORDER BY day
        """,
        (
            str(year),
            f"{month:02d}",
        )
    )


    rows = cursor.fetchall()


    connection.close()


    return {

        row["day"]:
            row["count"]

        for row in rows

    }


# ================================================================
# GET LEADS BY DATE
# ================================================================

def get_leads_by_date(
    selected_date
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT

            id,

            name,

            phone,

            email,

            qualification,

            career_goal,

            experience,

            course,

            training_mode,

            message,

            created_at

        FROM leads

        WHERE date(created_at)
            = ?

        ORDER BY id DESC
        """,
        (
            selected_date,
        )
    )


    rows = cursor.fetchall()


    connection.close()


    return [
        dict(row)
        for row in rows
    ]