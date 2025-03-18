import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_database_connection():
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "ml_project"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            port=os.getenv("DB_PORT", "5432")
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def init_database():
    conn = get_database_connection()
    if conn:
        cur = conn.cursor()

        # Create tables if they don't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                file_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                dataset_id INTEGER REFERENCES datasets(id),
                model_type VARCHAR(255) NOT NULL,
                model_path TEXT NOT NULL,
                metrics JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cur.close()
        conn.close()
