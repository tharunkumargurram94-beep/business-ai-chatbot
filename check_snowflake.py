import chromadb
from pathlib import Path

VECTORSTORE_DIR = Path("vectorstore")

client = chromadb.PersistentClient(
    path=str(VECTORSTORE_DIR)
)

collection = client.get_collection(
    name="business_documents"
)

results = collection.get(
    include=[
        "documents",
        "metadatas",
    ]
)

for document, metadata in zip(
    results["documents"],
    results["metadatas"],
):

    source = str(
        metadata.get(
            "source",
            ""
        )
    )

    title = str(
        metadata.get(
            "title",
            ""
        )
    )

    source_type = str(
        metadata.get(
            "source_type",
            ""
        )
    )

    combined = (
        source
        + " "
        + title
        + " "
        + document
    ).lower()

    if "snowflake" in combined:

        print("=" * 80)
        print("SOURCE:", source)
        print("SOURCE TYPE:", source_type)
        print("TITLE:", title)
        print("=" * 80)
        print(document[:5000])
        print()