import os
from astrapy import DataAPIClient
from dotenv import load_dotenv
load_dotenv()
def fetch_all_documents(collection):
    """Fetch all documents from the specified collection."""
    documents = []
    print("Debug: Starting to fetch documents...")
    for doc in collection.find():
        documents.append(doc)
        if len(documents) % 100 == 0:  # Print progress for every 100 documents
            print(f"Debug: Retrieved {len(documents)} documents so far...")
    print("Debug: Finished fetching documents.")
    return documents


def get_unique_combinations(documents):
    """Get unique combinations of 'element' and 'item' from the dataset."""
    unique_combinations = set()
    for doc in documents:
        metadata = doc.get("metadata", {})
        element = metadata.get("element")
        item = metadata.get("item")

        # Ensure element and item are strings before adding to the set
        if element is not None and item is not None:
            unique_combinations.add((str(element), str(item)))
    return list(unique_combinations)


def fetch_row_data(element, item, year):
    """
    Retrieve the data of a row for a specific element, item, and year.

    Args:
        collection: The database collection object.
        element (str): The element to search for.
        item (str): The item to search for.
        year (int): The year to search for.

    Returns:
        dict: The matched document if found, otherwise None.
    """
    query = {
        "metadata.element": element,
        "metadata.item": item,
        "metadata.year": year
    }
    # Initialize the Astra DB client
    ASTRA_DB_TOKEN= os.getenv("ASTRA_DB_TOKEN")
    client = DataAPIClient(ASTRA_DB_TOKEN)

    db = client.get_database_by_api_endpoint(
       "https://279507b1-1129-48a9-8c7b-9175be2856b5-us-east-2.apps.astra.datastax.com"
    )
    print("Debug: Connected to the database endpoint.")

    # Specify the collection name
    collection_name = "default_keyspaces"
    collection = db[collection_name]
    print(f"Debug: Querying for element={element}, item={item}, year={year}...")
    document = collection.find_one(query)
    if document:
        print("Debug: Document found.")
    else:
        print("Debug: No document matched the query.")
    return document


if __name__ == "__main__":
    try:
        # Initialize the Astra DB client
        ASTRA_DB_TOKEN= os.getenv("ASTRA_DB_TOKEN")
        client = DataAPIClient(ASTRA_DB_TOKEN)

        db = client.get_database_by_api_endpoint(
            "https://dfad9286-9c2b-4713-9c5a-709d9615cbcd-us-east-2.apps.astra.datastax.com"
        )
        print("Debug: Connected to the database endpoint.")

        # Specify the collection name
        collection_name = "default_keyspaces"
        collection = db[collection_name]

        # # Fetch all documents
        # documents = fetch_all_documents(collection)
        # print(f"Debug: Fetched {len(documents)} documents.")

        # # Get unique element-item combinations
        # unique_combinations = get_unique_combinations(documents)

        # # Print the results
        # print("Unique element-item combinations:")
        # for combination in unique_combinations:
        #     print(combination)

        # Fetch a specific row based on element, item, and year
        element = "Producing Animals/Slaughtered"
        item = "Sheep fat, unrendered"
        year = 1977
        row_data = fetch_row_data(element, item, year)

        if row_data:
            print("Row Data:")
            print(row_data)
        else:
            print("No matching data found for the given query.")

    except Exception as e:
        print(f"Error during processing: {e}")
