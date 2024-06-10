import sqlite3

import pandas as pd


def get_connector() -> sqlite3.Connection:
    """
    Connect to the project's SQLite database. Overwrite the default row factory that returns
    tuples with one that returns dictionaries.
    :return: connector object
    """
    conn = sqlite3.connect("data/leichte_sprache.db")
    # conn.row_factory = dict_factory
    return conn


def dict_factory(cursor: sqlite3.Connection, row):
    """
    By default, sqlite3 represents each row as a tuple. If a tuple does not suit your needs, you can use the sqlite3.Row class or a custom row_factory.
    While row_factory exists as an attribute both on the Cursor and the Connection, it is recommended to set Connection.row_factory, so all cursors created from the connection will use the same row factory.
    Row provides indexed and case-insensitive named access to columns, with minimal memory overhead and performance impact over a tuple. To use Row as a row factory, assign it to the row_factory attribute.
    Note: not currently used due to pandas bug: https://github.com/pandas-dev/pandas/issues/52437

    :param cursor: SQLite connection
    :param row: Dataset row
    :return: dictionary with the column name and the row value
    """
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def create_column_dict(
    col_name: str, col_dtype: str, pk: bool = False, not_null: bool = False
) -> dict:
    """Helper function to create a valid column dictionary used for creating a new table.

    :param col_name: name of the column
    :param col_dtype: valid SQL datatype
    :param pk: is this column the primary key? default: False
    :param not_null: must this column not be null? default: False
    :return: dictionary in standardized format
    """
    return {
        "name": col_name,
        "dtype": col_dtype,
        "primary_key": pk,
        "not_null": not_null,
    }


def create_table(name: str, columns: list[dict], dry_run: bool = False) -> None:
    """
    Create a table in the project's database. For each column, provide a dictionary
    in the following format:
        {
            "name": column_name,
            "dtype": column_data_type,
            "primary_key": False,
            "not_null": True,
        }
    dtype must be a string that can be parsed into an SQL datatype.

    :param name: table name
    :param columns: list of dictionaries in the format specified above, one dict per column
    :param dry_run: don't execute the statement, just print it
    """

    conn = get_connector()
    sql = f"CREATE TABLE IF NOT EXISTS {name} (\n"

    for col in columns:
        row_desc = f"{col['name']} {col['dtype']}"
        if col["primary_key"] is True:
            row_desc += " PRIMARY KEY"
        if col["not_null"] is True:
            row_desc += " NOT NULL"
        row_desc += ",\n"
        sql += row_desc
    sql = sql.strip(", \n")
    sql += ");"
    if not dry_run:
        conn.execute(sql)
        conn.commit()
        print(f"Created table {name} in project DB")
    else:
        print(sql)
    conn.close()
    return


def insert_rows(table_name: str, rows: list[dict], dry_run: bool = False):
    """Insert rows into a given table. To only print the generated SQL statement, set `dry_run=True`.

    :param table_name: name of the table in which to insert the data
    :param rows: list containing one dictionary per row. The dictionary keys must match the table's column names.
    :param dry_run: Don't run the command, just print the SQL statement. Defaults to False
    """
    # only use rows that share the same keys
    ref_keys = rows[0].keys()
    valid_rows = [r for r in rows if r.keys() == ref_keys]
    col_names = ", ".join([f":{k}" for k in ref_keys])

    conn = get_connector()
    sql = f"INSERT INTO {table_name} VALUES({col_names})"

    if not dry_run:
        conn.executemany(sql, valid_rows)
        conn.commit()
        print(f"Inserted {len(rows)} rows into table {table_name}")
    else:
        print(sql)

    conn.close()
    return


def ingest_csv(filepath: str, table_name: str):
    """Ingest a CSV file into the project's SQLite DB. If the table doesn't exist,
    a new one is created. Otherwise, the contents are appended to the existing table.
    In that case, make sure the columns match.

    :param filepath: path to the CSV file
    :param table_name: name of the new table
    """
    df = pd.read_csv(filepath)
    ingest_pandas(df, table_name)
    return


def ingest_pandas(df: pd.DataFrame, table_name: str, if_exists: str = "append"):
    """Ingest a pandas DataFrame into the project's SQLite DB. If the table doesn't exist,
    a new one is created. Otherwise, the behaviour is configured via the parameter
    `if_exists`.

    :param df: pandas dataframe
    :param table_name: name of the new table
    :param if_exists: {‘fail’, ‘replace’, ‘append’}, default ‘append’.
    """
    conn = get_connector()
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    return
