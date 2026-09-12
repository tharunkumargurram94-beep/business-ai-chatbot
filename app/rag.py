import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Please check your .env file."
    )


# ============================================================
# OPENAI
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# PATHS
# ============================================================

VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(VECTORSTORE_DIR)
)

try:

    collection = chroma_client.get_collection(
        name="business_documents"
    )

    print(
        "[RAG] Chroma collection loaded successfully."
    )

except Exception as error:

    collection = None

    print(
        f"[RAG] Chroma collection could not be loaded: {error}"
    )


# ============================================================
# COURSE ALIASES
# ============================================================

COURSE_ALIASES = {

    "Power BI": [
        "power bi",
        "powerbi",
        "power-bi",
    ],

    "Python": [
        "python",
        "python training",
        "python course",
    ],

    "Snowflake": [
        "snowflake",
        "snowflake training",
        "snowflake course",
    ],

    "SAP Security": [
        "sap security",
        "sap-security",
    ],

    "SAP GRC": [
        "sap grc",
        "sap-grc",
    ],

    "MS SQL": [
        "ms sql",
        "mssql",
        "sql server",
        "microsoft sql",
    ],

    "SSIS": [
        "ssis",
        "sql ssis",
    ],

    "Looker BI": [
        "looker bi",
        "looker",
        "lookerbi",
    ],

    "Fabric Administration": [
        "fabric administration",
        "fabric admin",
        "fabric administrator",
        "microsoft fabric administration",
    ],

    "Data Engineering": [
        "data engineering",
        "data engineer",
        "data engineering training",
    ],

    "Software Training": [
        "software training",
        "software course",
        "software courses",
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# COURSE DETECTION
# ============================================================

def detect_courses(text: str):

    if not text:
        return []

    normalized = normalize_text(text)

    found = []

    for course, aliases in COURSE_ALIASES.items():

        for alias in aliases:

            if normalize_text(alias) in normalized:

                if course not in found:
                    found.append(course)

                break

    return found


def detect_course(text: str):

    courses = detect_courses(text)

    if len(courses) == 1:
        return courses[0]

    return None


# ============================================================
# QUESTION TYPE
# ============================================================

def is_fee_question(question: str):

    q = normalize_text(question)

    keywords = [
        "fee",
        "fees",
        "price",
        "pricing",
        "cost",
        "costing",
        "charges",
        "course fee",
        "training fee",
        "how much",
        "how much does",
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


def is_duration_question(question: str):

    q = normalize_text(question)

    keywords = [
        "duration",
        "how long",
        "how many hours",
        "hours",
        "days",
        "months",
        "weeks",
        "course duration",
        "training duration",
        "time required",
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


def is_syllabus_question(question: str):

    q = normalize_text(question)

    keywords = [
        "syllabus",
        "curriculum",
        "topics",
        "modules",
        "course content",
        "training content",
        "what will i learn",
        "what do i learn",
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


def is_certification_question(question: str):

    q = normalize_text(question)

    keywords = [
        "certification",
        "certificate",
        "certified",
        "certification exam",
        "exam preparation",
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


# ============================================================
# FOLLOW-UP QUESTION DETECTION
# ============================================================

def is_followup_question(question: str):

    q = normalize_text(question)

    followups = [

        "what is the duration",
        "what's the duration",
        "duration?",
        "how long",
        "how many hours",

        "what is the fee",
        "what's the fee",
        "fees?",
        "fee?",

        "what is the syllabus",
        "what's the syllabus",
        "syllabus?",

        "is it online",
        "is this online",
        "online?",

        "what are the topics",
        "what topics",

        "what about certification",
        "is there certification",

        "what are the projects",
        "does it include projects",

        "what are the prerequisites",
        "what is required",

        "tell me more",
        "more details",
        "more information",
    ]

    return any(
        phrase in q
        for phrase in followups
    )


# ============================================================
# CONVERSATION CONTEXT
# ============================================================

def extract_course_from_history(
    conversation_history
):

    if not conversation_history:
        return None

    recent_messages = conversation_history[-12:]

    for item in reversed(recent_messages):

        if not isinstance(item, dict):
            continue

        text = item.get(
            "content",
            ""
        )

        if not text:
            continue

        detected = detect_courses(text)

        if len(detected) == 1:
            return detected[0]

    return None


# ============================================================
# LOCAL KNOWLEDGE SEARCH
# ============================================================

def search_local_knowledge(
    query: str,
    course: str | None = None,
    top_k: int = 20,
):

    if collection is None:
        return []

    try:

        search_query = query

        if course:

            search_query = (
                f"{course}. "
                f"{query}. "
                f"{course} training "
                f"syllabus duration fee "
                f"certification projects."
            )

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=search_query,
        )

        embedding = (
            embedding_response
            .data[0]
            .embedding
        )

        # ----------------------------------------------------
        # CHROMA SEARCH
        # ----------------------------------------------------

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        output = []

        for index, document in enumerate(
            documents
        ):

            metadata = {}

            if index < len(metadatas):

                metadata = (
                    metadatas[index]
                    or {}
                )

            distance = None

            if index < len(distances):
                distance = distances[index]

            output.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return output

    except Exception as error:

        print(
            f"[RAG] Search error: {error}"
        )

        return []


# ============================================================
# COURSE RESULT FILTER
# ============================================================

def filter_results(
    results,
    course: str | None = None,
    max_results: int = 10,
):

    if not results:
        return []

    # --------------------------------------------------------
    # GENERAL SEARCH
    # --------------------------------------------------------

    if not course:

        valid = []

        for result in results:

            distance = result.get(
                "distance"
            )

            if (
                distance is None
                or distance <= 1.5
            ):

                valid.append(result)

        return valid[:max_results]

    # --------------------------------------------------------
    # COURSE-SPECIFIC SEARCH
    # --------------------------------------------------------

    aliases = COURSE_ALIASES.get(
        course,
        [course],
    )

    aliases = [
        normalize_text(alias)
        for alias in aliases
    ]

    matched = []

    for result in results:

        document = result.get(
            "document",
            "",
        )

        metadata = result.get(
            "metadata",
            {},
        )

        source = str(
            metadata.get(
                "source",
                "",
            )
        )

        title = str(
            metadata.get(
                "title",
                "",
            )
        )

        combined = normalize_text(
            document
            + " "
            + source
            + " "
            + title
        )

        for alias in aliases:

            if alias in combined:

                matched.append(result)

                break

    return matched[:max_results]


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(results):

    if not results:
        return ""

    parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        document = result.get(
            "document",
            "",
        )

        metadata = result.get(
            "metadata",
            {},
        )

        title = metadata.get(
            "title",
            "",
        )

        source = metadata.get(
            "source",
            "",
        )

        header = (
            f"Knowledge Item {index}"
        )

        if title:
            header += f" | {title}"

        if source:
            header += f" | {source}"

        parts.append(
            f"{header}\n{document}"
        )

    return "\n\n".join(parts)


# ============================================================
# WEB SEARCH DECISION
# ============================================================

def should_use_web_search(
    question: str,
    course: str | None,
    local_results,
):

    q = normalize_text(question)

    # --------------------------------------------------------
    # COURSE-SPECIFIC VAMADEVA QUESTIONS
    # --------------------------------------------------------

    if course:

        return False

    # --------------------------------------------------------
    # VAMADEVA BUSINESS QUESTIONS
    # --------------------------------------------------------

    business_keywords = [

        "vamadeva",
        "training",
        "course",
        "courses",
        "fee",
        "fees",
        "duration",
        "batch",
        "batches",
        "admission",
        "enrollment",
        "placement",
        "job assistance",
        "live project",
        "live projects",
        "trainer",
        "class",
        "classes",
        "contact",
        "address",
        "syllabus",
        "curriculum",
        "certification",
        "certificate",
    ]

    if any(
        keyword in q
        for keyword in business_keywords
    ):

        return False

    # --------------------------------------------------------
    # CURRENT INFORMATION
    # --------------------------------------------------------

    current_keywords = [

        "latest",
        "current",
        "today",
        "recent",
        "2026",
        "news",
        "trend",
        "trends",
        "new version",
        "latest version",
    ]

    if any(
        keyword in q
        for keyword in current_keywords
    ):

        return True

    # --------------------------------------------------------
    # GENERAL TECHNOLOGY
    # --------------------------------------------------------

    general_keywords = [

        "what is",
        "how does",
        "how do",
        "difference between",
        "compare",
        "advantages",
        "disadvantages",
        "career",
        "salary",
        "job market",
        "industry",
        "technology",
    ]

    if any(
        keyword in q
        for keyword in general_keywords
    ):

        return True

    return False


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the official AI assistant for Vamadeva Techno Solutions.

You help website visitors understand:

- Courses
- Training programs
- Course syllabus
- Course duration
- Fees when confirmed
- Certification
- Projects
- Training modes
- Career paths
- Job assistance
- Company services
- General technology questions


============================================================
COURSE CONTEXT
============================================================

The application may provide a CURRENT COURSE.

If CURRENT COURSE exists, it is the active course/topic.

Example:

CURRENT COURSE:
Data Engineering

USER QUESTION:
What is the duration to learn this course?

You MUST understand "this course" as Data Engineering.

DO NOT ask the user to select the course again.

Example:

CURRENT COURSE:
Power BI

USER QUESTION:
What is the fee?

Answer specifically for Power BI.

Never mix information from different courses.


============================================================
CONVERSATION
============================================================

Previous conversation may be provided.

Use it to understand references such as:

- this course
- that course
- it
- this training
- the above course
- the program
- that training

If the active course is already known, continue using it.


============================================================
VAMADEVA DATA
============================================================

Private Vamadeva knowledge is authoritative for
Vamadeva-specific information.

Use it for:

- Course details
- Syllabus
- Duration
- Fees
- Training modes
- Projects
- Certification
- Job assistance
- Company services

Do not invent Vamadeva-specific information.


============================================================
FEES
============================================================

NEVER invent a fee.

If the supplied Vamadeva information contains a confirmed
fee, provide it.

If no confirmed fee is available, say:

"The current fee is not available in the information I have.
Please contact Vamadeva Techno Solutions for the latest fee
structure."

Do not estimate a fee.


============================================================
DURATION
============================================================

Never invent course duration.

Use supplied Vamadeva information when available.

If it is not available, clearly say that the current duration
is not available in the information.


============================================================
WEB INFORMATION
============================================================

Public web information can be used for general technology
questions and current information.

Do not use public web information to invent Vamadeva-specific
fees, promises, or business information.


============================================================
RESPONSE STYLE
============================================================

Be professional, friendly, and helpful.

Use simple language that a student or website visitor can
understand.

Keep answers reasonably concise.

Use Markdown formatting when useful.

For headings use:

### Heading

For important information use:

**Important**

For lists use:

- Point 1
- Point 2
- Point 3

IMPORTANT FORMATTING RULES:

1. Always use normal spaces between words.
2. Never join two words together.
3. Never split a word into separate pieces.
4. Do not insert spaces inside words.
5. Use normal punctuation spacing.
6. Put a space after commas when appropriate.
7. Put a space after periods when another sentence follows.
8. Keep technical names correctly spelled.
9. Keep names such as Python, NumPy, Pandas, Matplotlib,
   Django, Flask, Power BI, Snowflake, SQL Server, SAP GRC,
   SAP Security, and Microsoft Fabric correctly formatted.
10. Do not use unnecessary blank lines.
11. Do not repeat the user's question unnecessarily.
12. Do not create overly long introductions.
13. Give the answer directly.

Correct:

Python is known for its readability.

Correct:

NumPy, Pandas, and Matplotlib are commonly used libraries.

Correct:

Python is suitable for beginners and experienced developers.

Incorrect:

it s readability

Incorrect:

in telligence

Incorrect:

in cluding

Incorrect:

object-or iented

Incorrect:

platform-in dependent

Incorrect:

suitablefor

Incorrect:

forbeginners


============================================================
NO SOURCE DISPLAY
============================================================

Never show:

- URLs
- source lists
- citations
- internal document names
- ChromaDB
- embeddings
- RAG
- database information
- internal system information

The user should experience a normal professional chatbot.


============================================================
IMPORTANT
============================================================

Never fabricate information.

If information is unavailable, say so honestly.

For Vamadeva-specific questions, prefer the supplied
Vamadeva knowledge over general knowledge.

For general technology questions, provide a clear,
accurate explanation.

Never mix unrelated course information.

Never claim that Vamadeva provides something unless the
supplied information supports it.
"""


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_rag_answer(
    question: str,
    current_course: str | None = None,
    conversation_history: list[dict] | None = None,
):

    try:

        if not question:

            return {
                "answer": "Please enter a question.",
                "course": current_course,
            }

        conversation_history = (
            conversation_history or []
        )

        # ====================================================
        # STEP 1
        # ACTUAL QUESTION
        # ====================================================

        actual_question = question.strip()

        # ====================================================
        # STEP 2
        # DETECT COURSE FROM CURRENT QUESTION
        # ====================================================

        detected_courses = detect_courses(
            actual_question
        )

        if len(detected_courses) == 1:

            current_course = detected_courses[0]

        # ====================================================
        # STEP 3
        # USE CONVERSATION HISTORY
        # ====================================================

        if not current_course:

            current_course = (
                extract_course_from_history(
                    conversation_history
                )
            )

        # ====================================================
        # STEP 4
        # HANDLE FOLLOW-UP QUESTIONS
        # ====================================================

        if (
            not current_course
            and is_followup_question(
                actual_question
            )
        ):

            return {
                "answer": (
                    "Sure. Which course would you like "
                    "to know about? For example, Power BI, "
                    "Python, Snowflake, Data Engineering, "
                    "SAP Security, SAP GRC, MS SQL, SSIS, "
                    "Looker BI, or Fabric Administration."
                ),
                "course": None,
            }

        # ====================================================
        # STEP 5
        # LOCAL RAG
        # ====================================================

        local_results = search_local_knowledge(
            actual_question,
            course=current_course,
            top_k=20,
        )

        filtered_results = filter_results(
            local_results,
            course=current_course,
            max_results=10,
        )

        context = build_context(
            filtered_results
        )

        # ====================================================
        # STEP 6
        # WEB SEARCH
        # ====================================================

        use_web = should_use_web_search(
            actual_question,
            current_course,
            filtered_results,
        )

        # ====================================================
        # STEP 7
        # PREVIOUS CONVERSATION
        # ====================================================

        history_text = ""

        if conversation_history:

            recent_history = (
                conversation_history[-10:]
            )

            history_parts = []

            for item in recent_history:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                role = item.get(
                    "role",
                    "",
                )

                content = item.get(
                    "content",
                    "",
                )

                if not content:
                    continue

                history_parts.append(
                    f"{role.upper()}: {content}"
                )

            if history_parts:

                history_text = (
                    "RECENT CONVERSATION:\n"
                    + "\n".join(history_parts)
                )

        # ====================================================
        # STEP 8
        # BUILD PROMPT
        # ====================================================

        prompt_parts = []

        if current_course:

            prompt_parts.append(
                "CURRENT COURSE:\n"
                + current_course
            )

        if history_text:

            prompt_parts.append(
                history_text
            )

        if context:

            prompt_parts.append(
                "PRIVATE VAMADEVA KNOWLEDGE:\n"
                + context
            )

        prompt_parts.append(
            "USER QUESTION:\n"
            + actual_question
        )

        user_prompt = "\n\n".join(
            prompt_parts
        )

        # ====================================================
        # STEP 9
        # TOOLS
        # ====================================================

        tools = []

        if use_web:

            tools.append(
                {
                    "type": "web_search"
                }
            )

        # ====================================================
        # STEP 10
        # OPENAI RESPONSE
        # ====================================================

        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            tools=tools,
        )

        answer = response.output_text

        if not answer:

            answer = (
                "I'm sorry, I couldn't generate an answer "
                "right now. Please try again."
            )

        # ====================================================
        # STEP 11
        # CLEAN RESPONSE
        # ====================================================

        answer = clean_answer(
            answer
        )

        # ====================================================
        # STEP 12
        # RETURN COURSE CONTEXT
        # ====================================================

        return {
            "answer": answer,
            "course": current_course,
        }

    except Exception as error:

        print(
            f"[RAG] ERROR: {error}"
        )

        return {
            "answer": (
                "I'm sorry, I'm having trouble processing "
                "that request right now. Please try again."
            ),
            "course": current_course,
        }


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer: str):

    if not answer:
        return ""

    # --------------------------------------------------------
    # REMOVE CITATION MARKERS
    # --------------------------------------------------------

    answer = re.sub(
        r"\[\d+\]",
        "",
        answer,
    )

    answer = re.sub(
        r"【.*?】",
        "",
        answer,
    )

    # --------------------------------------------------------
    # REMOVE SOURCE HEADING
    # --------------------------------------------------------

    answer = re.sub(
        r"(?im)^sources?:\s*$",
        "",
        answer,
    )

    # --------------------------------------------------------
    # CONVERT MARKDOWN LINKS TO TEXT
    # --------------------------------------------------------

    answer = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        answer,
    )

    # --------------------------------------------------------
    # REMOVE RAW URLS
    # --------------------------------------------------------

    answer = re.sub(
        r"https?://\S+",
        "",
        answer,
    )

    # --------------------------------------------------------
    # FIX PUNCTUATION SPACING
    # --------------------------------------------------------

    # Remove spaces before punctuation.
    answer = re.sub(
        r"[ \t]+([,.;:!?])",
        r"\1",
        answer,
    )

    # Add a space after commas when missing.
    answer = re.sub(
        r",(?=[A-Za-z0-9])",
        ", ",
        answer,
    )

    # Add a space after semicolons when missing.
    answer = re.sub(
        r";(?=[A-Za-z0-9])",
        "; ",
        answer,
    )

    # Add a space after colons when missing.
    answer = re.sub(
        r":(?=[A-Za-z0-9])",
        ": ",
        answer,
    )

    # --------------------------------------------------------
    # FIX ONLY VERY CLEAR WORD-JOINING ERRORS
    # --------------------------------------------------------
    #
    # DO NOT use generic rules such as:
    #
    # "it" -> "it "
    # "in" -> "in "
    # "for" -> "for "
    #
    # because those can corrupt valid words such as:
    #
    # its
    # intelligence
    # including
    # industry
    #
    # Only clearly identifiable formatting mistakes are fixed.

    replacements = {

        "suitablefor": "suitable for",
        "forbeginners": "for beginners",

        "NumPy,Pandas": "NumPy, Pandas",
        "numpy,pandas": "NumPy, Pandas",

        "Pandas,Matplotlib": "Pandas, Matplotlib",
        "pandas,matplotlib": "Pandas, Matplotlib",

        "andPandas": "and Pandas",
        "andNumPy": "and NumPy",
        "andMatplotlib": "and Matplotlib",

        "MachineLearning": "Machine Learning",
        "DataScience": "Data Science",
        "WebDevelopment": "Web Development",
        "SoftwareDevelopment": "Software Development",
        "SystemAdministration": "System Administration",
        "DatabaseProgramming": "Database Programming",
        "NetworkProgramming": "Network Programming",

        "OpenSource": "Open Source",
        "GeneralPurpose": "General-Purpose",

        "object-or iented": "object-oriented",
        "platform-in dependent": "platform-independent",

        "be ginners": "beginners",
        "in telligence": "intelligence",
        "in cluding": "including",
    }

    for old, new in replacements.items():

        answer = answer.replace(
            old,
            new,
        )

    # --------------------------------------------------------
    # CLEAN MULTIPLE SPACES
    # --------------------------------------------------------

    answer = re.sub(
        r"[ \t]{2,}",
        " ",
        answer,
    )

    # --------------------------------------------------------
    # CLEAN SPACES ON EMPTY LINES
    # --------------------------------------------------------

    answer = re.sub(
        r"\n[ \t]+",
        "\n",
        answer,
    )

    # --------------------------------------------------------
    # LIMIT EXCESSIVE BLANK LINES
    # --------------------------------------------------------

    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer,
    )

    # --------------------------------------------------------
    # CLEAN MARKDOWN BULLET SPACING
    # --------------------------------------------------------

    answer = re.sub(
        r"(?m)^[ \t]+([-*])",
        r"\1",
        answer,
    )

    # --------------------------------------------------------
    # FINAL CLEANUP
    # --------------------------------------------------------

    return answer.strip()