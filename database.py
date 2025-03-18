import sqlite3
import os
from datetime import datetime

def get_database_connection():
    try:
        # Create a connection to SQLite database
        conn = sqlite3.connect('ml_project.db')
        return conn
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                file_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                dataset_id INTEGER,
                model_type TEXT NOT NULL,
                model_path TEXT NOT NULL,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            )
        """)

        conn.commit()
        cur.close()
        conn.close()
