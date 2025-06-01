import psycopg2
from pgvector.psycopg2 import register_vector
from .config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def get_db_connection(db_name):
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        register_vector(conn)
        print("Connected to DB and registered pgvector.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"DB connection failed: {e}")
        raise