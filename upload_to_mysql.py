import os
import pandas as pd
import mysql.connector
from mysql.connector import Error
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_upload.log'),
        logging.StreamHandler()
    ]
)

# Database configuration
DB_CONFIG = {
    'host': '68.178.150.182',  # Replace with your GoDaddy MySQL host
    'user': 'kc099',  # Replace with your username
    'password': 'Roboworks23!',  # Replace with your password
    'database': 'ecommerce'  # Replace with your database name
}

def create_connection():
    """Create a database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            logging.info("Successfully connected to MySQL database")
            return connection
    except Error as e:
        logging.error(f"Error connecting to MySQL database: {e}")
        raise

def create_table_from_csv(connection, csv_file, table_name):
    """Create a table based on CSV structure"""
    try:
        # Read first few rows to infer schema
        df = pd.read_csv(csv_file, nrows=5)
        
        # Generate CREATE TABLE statement
        columns = []
        for col_name, dtype in df.dtypes.items():
            if 'int' in str(dtype):
                sql_type = 'INT'
            elif 'float' in str(dtype):
                sql_type = 'FLOAT'
            elif 'datetime' in str(dtype):
                sql_type = 'DATETIME'
            else:
                # For string columns, check max length
                max_length = df[col_name].astype(str).str.len().max()
                sql_type = f'VARCHAR({max(max_length * 2, 255)})'  # Double the max length for safety
            
            # Clean column name (remove special characters)
            clean_col_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in col_name)
            columns.append(f"`{clean_col_name}` {sql_type}")
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            {', '.join(columns)}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        cursor = connection.cursor()
        cursor.execute(create_table_sql)
        connection.commit()
        logging.info(f"Created table {table_name}")
        cursor.close()
        
    except Error as e:
        logging.error(f"Error creating table {table_name}: {e}")
        raise

def upload_csv_to_table(connection, csv_file, table_name, chunk_size=10000):
    """Upload CSV data to MySQL table in chunks"""
    try:
        cursor = connection.cursor()
        total_rows = 0
        start_time = time.time()
        
        # Read and upload in chunks
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            # Clean column names
            chunk.columns = [''.join(c if c.isalnum() or c == '_' else '_' for c in col) for col in chunk.columns]
            
            # Replace NaN values with None (which becomes NULL in MySQL)
            chunk = chunk.replace({pd.NA: None, pd.NaT: None})
            chunk = chunk.where(pd.notnull(chunk), None)
            
            # Convert DataFrame to list of tuples
            values = [tuple(None if pd.isna(x) else x for x in row) for row in chunk.values]
            
            # Generate placeholders for SQL query
            placeholders = ', '.join(['%s'] * len(chunk.columns))
            columns = ', '.join([f'`{col}`' for col in chunk.columns])
            
            # Prepare and execute insert query
            insert_query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
            cursor.executemany(insert_query, values)
            connection.commit()
            
            total_rows += len(chunk)
            elapsed_time = time.time() - start_time
            logging.info(f"Uploaded {total_rows} rows to {table_name} in {elapsed_time:.2f} seconds")
        
        cursor.close()
        logging.info(f"Successfully uploaded {total_rows} rows to {table_name}")
        
    except Error as e:
        logging.error(f"Error uploading data to {table_name}: {e}")
        raise

def process_csv_files():
    """Process all CSV files in the ecommerce_data1 directory"""
    csv_dir = 'ecommerce_data1'
    connection = None
    
    try:
        connection = create_connection()
        
        for csv_file in os.listdir(csv_dir):
            if csv_file.endswith('browsing_sessions.csv'):
                file_path = os.path.join(csv_dir, csv_file)
                table_name = os.path.splitext(csv_file)[0].lower()
                
                logging.info(f"Processing {csv_file}")
                
                # Create table
                create_table_from_csv(connection, file_path, table_name)
                
                # Upload data
                upload_csv_to_table(connection, file_path, table_name)
                
    except Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if connection and connection.is_connected():
            connection.close()
            logging.info("Database connection closed")

def count_csv_columns(csv_file):
    """Count the number of columns in a CSV file"""
    try:
        df = pd.read_csv(csv_file)
        num_columns = len(df.columns)
        column_names = df.columns.tolist()
        logging.info(f"File: {csv_file}")
        logging.info(f"Number of columns: {num_columns}")
        logging.info(f"Column names: {column_names}")
        return num_columns, column_names
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        raise

if __name__ == "__main__":
    # Example usage to count columns
    csv_file = "ecommerce_data/browsing_sessions.csv"
    if os.path.exists(csv_file):
        pass
        # print(count_csv_columns(csv_file))
    
    logging.info("Starting CSV to MySQL upload process")
    process_csv_files()
    logging.info("Upload process completed") 