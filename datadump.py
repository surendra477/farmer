from astrapy import DataAPIClient
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from dotenv import load_dotenv
import os
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
device = torch.device("cpu")  # Force CPU usage

class AstraVectorDB:
    def __init__(self, token, api_endpoint):
        self.client = DataAPIClient(token)
        self.db = self.client.get_database_by_api_endpoint(api_endpoint)
        self.model = SentenceTransformer('all-MiniLM-L6-v2',device=device)
    
    def validate_dataframe(self, df):
        """Validate and clean the dataframe"""
        # Print column names for debugging
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Expected columns
        expected_columns = [
            "Domain Code", "Domain", "Area Code", "Area", 
            "Element Code", "Element", "Item Code", "Item", 
            "Year Code", "Year", "Unit", "Value", "Flag", 
            "Flag Description"
        ]
        
        # Check for missing columns
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean the dataframe
        df_clean = df.copy()
        
        # Convert to appropriate types with error handling
        try:
            df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
            df_clean['Value'] = pd.to_numeric(df_clean['Value'], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting numeric columns: {str(e)}")
        
        # Fill NA/NaN values
        df_clean = df_clean.fillna({
            'Domain Code': 'unknown',
            'Domain': 'unknown',
            'Area Code': 'unknown',
            'Area': 'unknown',
            'Element Code': 'unknown',
            'Element': 'unknown',
            'Item Code': 'unknown',
            'Item': 'unknown',
            'Unit': 'unknown',
            'Flag': '',
            'Flag Description': ''
        })
        
        return df_clean

    def process_and_insert_data(self, csv_path, collection_name, batch_size=100):
        try:
            # Read CSV file
            logger.info(f"Reading CSV file from {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Original columns: {df.columns.tolist()}")
            
            # Validate and clean data
            df_clean = self.validate_dataframe(df)
            total_rows = len(df_clean)
            logger.info(f"Total rows to process: {total_rows}")
            
            # Get collection
            collection = self.db[collection_name]
            
            # Process in batches
            for start_idx in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df_clean.iloc[start_idx:end_idx]
                
                batch_documents = []
                for _, row in batch_df.iterrows():
                    try:
                        # Create text for embedding
                        text_content = (
                            f"In {int(row['Year'])}, {str(row['Area'])} reported "
                            f"{float(row['Value'])} {str(row['Unit'])} for {str(row['Item'])} "
                            f"under {str(row['Element'])}. Domain: {str(row['Domain'])}"
                        )
                        
                        # Generate embedding
                        embedding = self.model.encode(text_content).tolist()
                        
                        # Create document
                        document = {
                            "text_content": text_content,
                            "embedding": embedding,
                            "metadata": {
                                "domain_code": str(row['Domain Code']),
                                "domain": str(row['Domain']),
                                "area_code": str(row['Area Code']),
                                "area": str(row['Area']),
                                "element_code": str(row['Element Code']),
                                "element": str(row['Element']),
                                "item_code": str(row['Item Code']),
                                "item": str(row['Item']),
                                "year": int(row['Year']),
                                "unit": str(row['Unit']),
                                "value": float(row['Value']),
                                "flag": str(row['Flag']),
                                "flag_description": str(row['Flag Description'])
                            }
                        }
                        batch_documents.append(document)
                        
                    except Exception as e:
                        logger.error(f"Error processing row: {str(e)}\nRow data: {row.to_dict()}")
                        continue
                
                # Insert batch
                if batch_documents:
                    try:
                        collection.insert_many(batch_documents)
                        logger.info(f"Inserted batch of {len(batch_documents)} documents")
                    except Exception as e:
                        logger.error(f"Error inserting batch: {str(e)}")
                
            logger.info("Data ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process CSV file: {str(e)}")
            raise

def main():
    # Configuration
    config = {
        'token': os.getenv("ASTRA_DB_TOKEN"),
        'api_endpoint': os.getenv('ASTRA_DB_ENDPOINT'),
        'collection_name': 'default_keyspaces',
        'csv_path': 'datafarmer.csv'  # Update this path
    }
    
    try:
        # Initialize Astra Vector DB connection
        astra_db = AstraVectorDB(config['token'], config['api_endpoint'])
        
        # Process and insert data
        astra_db.process_and_insert_data(
            config['csv_path'],
            config['collection_name']
        )
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()