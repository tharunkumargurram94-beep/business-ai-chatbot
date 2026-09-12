import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

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

BASE_DIR = Path(__file__).resolve().parent.parent

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

    # Search recent messages first
    recent_messages = conversation_history[-12:]

    for item in reversed(recent_messages):

        if not isinstance(item, dict):
            continue

        text = item.get(
            "content",
            ""
        )

        role = item.get(
            "role",
            ""
        )

        if not text:
            continue

        detected = detect_course(text)

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

        for index, document in enumerate(documents):

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

Be professional and friendly.

Use simple language.

Use Markdown formatting:

### Heading

**Important**

- Point 1
- Point 2

Keep answers reasonably concise.

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
        # ====================================================

        actual_question = question.strip()

        # ====================================================
        # STEP 2
        # DETECT COURSE FROM CURRENT QUESTION
        # ====================================================

        detected_courses = detect_courses(
            actual_question
        )

        # If user explicitly mentions one course,
        # that course becomes active.

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

                # Don't expose internal metadata
                # to the model unnecessarily.

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
                "I'm sorry, I’m having trouble processing "
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

    # Remove citation markers
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

    # Remove markdown source heading
    answer = re.sub(
        r"(?im)^sources?:\s*$",
        "",
        answer,
    )

    # Convert markdown links to text
    answer = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        answer,
    )

    # Remove raw URLs
    answer = re.sub(
        r"https?://\S+",
        "",
        answer,
    )

    # Remove excessive blank lines
    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer,
    )

    return answer.strip()