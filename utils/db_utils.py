# utils/db_utils.py
import sqlite3
def save_to_db(df, db_path, table_name):
    """
    Save the DataFrame to a SQLite3 database table.
    - If table exists, it will be replaced.
    """
    
    
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists='append', index=False)#if table already exists,do not overwrite it
    print(f"[INFO] Saved data to table: {table_name}")

def drop_table(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        print(f"[INFO] Dropped table: {table_name}")

def drop_all_tables(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get list of all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            # Properly quote the table name in SQL
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            print(f"[INFO] Dropped table: '{table_name}'")
if __name__=="__main__":
    drop_all_tables(db_path=r"D:\learning_alpha_edge\db\market_data.db")