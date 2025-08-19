from sqlalchemy import create_engine, text,Engine
import pandas as pd


def get_pg_engine(user, password, host, port, dbname)->Engine:
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")


def save_to_db(df:pd.DataFrame, engine:Engine, schema:str, table_name:str):
    """
    Save DataFrame to a PostgreSQL table inside a specific schema.
    - Appends if table exists, creates otherwise.
    """
    with engine.begin() as conn:
        df.to_sql(
        name=table_name,
        con=conn,
        schema=schema,
        if_exists='append',
        index=False
    )
    
    print(f"[INFO] Saved data to {schema}.{table_name}")

def load_data(schema:str,table_name:str,engine)->pd.DataFrame:
    query=f'SELECT * FROM "{schema}"."{table_name}" ORDER BY datetime ASC;'
    df=pd.read_sql(query,engine)
    df["datetime"]=pd.to_datetime(df["datetime"])
    return df


def drop_table(engine:Engine, schema, table_name):
    """
    Drop a specific table in a given schema.
    """
    with engine.connect() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{table_name}" CASCADE'))
        print(f"[INFO] Dropped table: {schema}.{table_name}")


def drop_all_tables(engine:Engine, schema):
    """
    Drop all tables within a given schema.
    """
    with engine.connect() as conn: # Use engine.begin() for auto-committing transactions
        result = conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = :schema"
        ), {"schema": schema})
        tables = result.fetchall()

        for (table_name,) in tables:
            conn.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{table_name}" CASCADE'))
            print(f"[INFO] Dropped table: {schema}.{table_name}")
            conn.commit()


def ensure_schema_exists(engine:Engine, schema):
    """
    Create the schema if it doesn't exist.
    """
    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        print(f"[INFO] Ensured schema exists: {schema}")


# Example usage
if __name__ == "__main__":
    engine = get_pg_engine(
        user="postgres",
        password="Afridi11",
        host="localhost",
        port="5432",
        dbname="db"
    )
    schema = ["public","signals","ledger"]
    for schema_name in schema:
        drop_all_tables(engine, schema_name)


    