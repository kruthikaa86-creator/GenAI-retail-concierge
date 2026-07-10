from langchain_core.documents import Document

from src.data.loader import load_all_datasets


def create_product_documents(products):
    """
    Create one aggregated product document per product ID.

    Stable attributes are represented directly.
    Conflicting attributes are preserved as observed values
    from the raw product records.
    """

    documents = []

    grouped_products = products.groupby(
        "product_id",
        sort=False,
    )

    for product_id, product_records in grouped_products:

        first_record = product_records.iloc[0]

        categories = sorted(
            product_records["category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        subcategories = sorted(
            product_records["subcategory"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        statuses = sorted(
            product_records["status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        minimum_stock = int(
            product_records["stock"].min()
        )

        maximum_stock = int(
            product_records["stock"].max()
        )

        minimum_rating = float(
            product_records["avg_rating"].min()
        )

        maximum_rating = float(
            product_records["avg_rating"].max()
        )

        content = f"""
Product Name: {first_record['name']}
Brand: {first_record['brand']}
Price: {first_record['price']}
Observed Categories: {", ".join(categories)}
Observed Subcategories: {", ".join(subcategories)}
Observed Statuses: {", ".join(statuses)}
Observed Stock Range: {minimum_stock} to {maximum_stock}
Observed Rating Range: {minimum_rating} to {maximum_rating}
Description: {first_record['description']}
""".strip()

        document = Document(
            page_content=content,
            metadata={
                "source": "products",
                "product_id": product_id,
                "image_url": first_record["image_url"],
                "raw_record_count": int(
                    len(product_records)
                ),
            },
        )

        documents.append(document)

    return documents
def create_review_documents(reviews):
    """
    Create one aggregated review document per product ID.
    """

    documents = []

    grouped_reviews = reviews.groupby("product_id")

    for product_id, product_reviews in grouped_reviews:

        review_count = len(product_reviews)

        average_rating = round(
    float(product_reviews["rating"].mean()),
    2,
)

        recent_reviews = (
            product_reviews
            .sort_values(
                "review_date",
                ascending=False,
            )
            .head(3)
        )

        review_texts = []

        for _, review in recent_reviews.iterrows():

            review_texts.append(
                f"Rating {review['rating']}: "
                f"{review['review_text']}"
            )

        review_summary = "\n".join(review_texts)

        content = f"""
Product ID: {product_id}
Total Reviews: {review_count}
Average Review Rating: {average_rating}
Recent Customer Reviews:
{review_summary}
""".strip()

        document = Document(
            page_content=content,
            metadata={
                "source": "product_reviews",
                "product_id": product_id,
                "review_count": review_count,
                "average_rating": average_rating,
            },
        )

        documents.append(document)

    return documents


def create_policy_documents(policies):
    """
    Convert policy JSON records into LangChain documents.
    """

    documents = []

    for policy in policies:

        content = f"""
Policy Question: {policy['question']}
Policy Answer: {policy['answer']}
Category: {policy['category']}
Policy Source: {policy['source']}
""".strip()

        document = Document(
            page_content=content,
            metadata={
                "source": "operations_policies",
                "policy_id": policy["id"],
                "category": policy["category"],
                "policy_source": policy["source"],
            },
        )

        documents.append(document)

    return documents


def create_rag_documents():
    """
    Create all documents required for the RAG vector store.
    """

    datasets = load_all_datasets()

    product_documents = create_product_documents(
        datasets["products"]
    )

    review_documents = create_review_documents(
        datasets["product_reviews"]
    )

    policy_documents = create_policy_documents(
        datasets["operations_policies"]
    )

    all_documents = (
        product_documents
        + review_documents
        + policy_documents
    )

    return {
        "products": product_documents,
        "reviews": review_documents,
        "policies": policy_documents,
        "all": all_documents,
    }


if __name__ == "__main__":

    documents = create_rag_documents()

    print("\nRAG DOCUMENT CREATION COMPLETE\n")

    print(
        f"Product documents: "
        f"{len(documents['products'])}"
    )

    print(
        f"Review documents: "
        f"{len(documents['reviews'])}"
    )

    print(
        f"Policy documents: "
        f"{len(documents['policies'])}"
    )

    print(
        f"Total documents: "
        f"{len(documents['all'])}"
    )

    print("\nSAMPLE PRODUCT DOCUMENT:")
    print(
        documents["products"][0].page_content
    )

    print("\nPRODUCT METADATA:")
    print(
        documents["products"][0].metadata
    )

    print("\nSAMPLE REVIEW DOCUMENT:")
    print(
        documents["reviews"][0].page_content
    )

    print("\nREVIEW METADATA:")
    print(
        documents["reviews"][0].metadata
    )

    print("\nSAMPLE POLICY DOCUMENT:")
    print(
        documents["policies"][0].page_content
    )

    print("\nPOLICY METADATA:")
    print(
        documents["policies"][0].metadata
    )