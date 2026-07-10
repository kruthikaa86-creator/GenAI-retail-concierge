from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.config import BASE_DIR
from src.retrieval.documents import create_rag_documents
from src.retrieval.embeddings import get_embedding_model


VECTORSTORE_DIR = BASE_DIR / "vectorstore"

PRODUCT_INDEX_DIR = VECTORSTORE_DIR / "products"
REVIEW_INDEX_DIR = VECTORSTORE_DIR / "reviews"
POLICY_INDEX_DIR = VECTORSTORE_DIR / "policies"


def build_and_save_index(
    documents,
    index_directory: Path,
    index_name: str,
) -> None:
    """
    Build and save one FAISS vector index.
    """

    if not documents:
        raise ValueError(
            f"No documents supplied for {index_name} index."
        )

    print("\n" + "=" * 60)
    print(f"BUILDING INDEX: {index_name.upper()}")
    print("=" * 60)

    print(f"Documents: {len(documents)}")

    embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model,
    )

    index_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(
        str(index_directory)
    )

    print(
        f"{index_name} index saved to: "
        f"{index_directory}"
    )


def build_vector_stores() -> None:
    """
    Build all FAISS indexes.
    """

    documents = create_rag_documents()

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    build_and_save_index(
        documents=documents["products"],
        index_directory=PRODUCT_INDEX_DIR,
        index_name="products",
    )

    build_and_save_index(
        documents=documents["reviews"],
        index_directory=REVIEW_INDEX_DIR,
        index_name="reviews",
    )

    build_and_save_index(
        documents=documents["policies"],
        index_directory=POLICY_INDEX_DIR,
        index_name="policies",
    )

    print("\nVECTOR STORE BUILD COMPLETE")


def rebuild_product_index() -> None:
    """
    Rebuild only the product FAISS index.
    """

    print("\nLOADING PRODUCT DOCUMENTS")

    documents = create_rag_documents()

    product_documents = documents["products"]

    print(
        f"Product documents loaded: "
        f"{len(product_documents)}"
    )

    build_and_save_index(
        documents=product_documents,
        index_directory=PRODUCT_INDEX_DIR,
        index_name="products",
    )

    print("\nPRODUCT INDEX REBUILD COMPLETE")


def load_index(
    index_directory: Path,
) -> FAISS:
    """
    Load a saved FAISS index.
    """

    embedding_model = get_embedding_model()

    if not index_directory.exists():
        raise FileNotFoundError(
            f"Vector index not found: {index_directory}"
        )

    vector_store = FAISS.load_local(
        str(index_directory),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def verify_saved_indexes() -> None:
    """
    Verify that all saved indexes can be loaded.
    """

    indexes = {
        "products": PRODUCT_INDEX_DIR,
        "reviews": REVIEW_INDEX_DIR,
        "policies": POLICY_INDEX_DIR,
    }

    print("\nVERIFYING SAVED INDEXES\n")

    for index_name, index_directory in indexes.items():

        vector_store = load_index(
            index_directory
        )

        document_count = len(
            vector_store.index_to_docstore_id
        )

        print(
            f"{index_name}: "
            f"{document_count} documents loaded"
        )


if __name__ == "__main__":

    build_vector_stores()

    verify_saved_indexes()