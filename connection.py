from google.cloud.sql.connector import Connector
import sqlalchemy

# Initialize the Connector object
connector = Connector()

# Function to return a DB-API connection
def getconn():
    return connector.connect(
        "unique-decker-479723-e8:us-central1:attendance-db",  # Cloud SQL instance connection name
        "pymysql",
        user="root",           # Your DB user
        password="Password@123",  # Your DB password
        db="attendance-system"     # Your database name
    )

# Initialize SQLAlchemy engine using the connector
engine = sqlalchemy.create_engine(
    "mysql+pymysql://",  # Note: empty credentials because connector handles it
    creator=getconn,
    pool_size=5,
    max_overflow=2,
    pool_timeout=30,
    pool_recycle=1800,
)
