import pytest
from sqlite3 import OperationalError, IntegrityError

from leichte_sprache.utils.db_utils import (
    create_column_dict,
    create_table,
    get_connector,
    insert_rows,
)


@pytest.fixture(scope="function", autouse=True)
def run_around_tests(conn):
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want

    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    # drop all tables in the test DB to have a clean state for every test
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_schema WHERE type='table';")
    tables = cur.fetchall()

    for (table,) in tables:
        cur.execute(f"DROP TABLE {table}")
    cur.close()
    return


@pytest.mark.parametrize(
    "col_name, col_dtype, pk, not_null",
    (["id", "int", True, True], ["foo", "text", False, False]),
)
def test_create_column_dict(col_name, col_dtype, pk, not_null):
    col_dict = create_column_dict(col_name, col_dtype, pk, not_null)
    assert isinstance(col_dict, dict)
    assert col_dict["name"]
    assert col_dict["dtype"]
    assert col_dict["primary_key"] is not None
    assert col_dict["not_null"] is not None
    return


@pytest.mark.parametrize(
    "col_dict, dry_run, expected",
    (
        [
            {
                "col_name": "id",
                "col_dtype": "int",
                "pk": True,
                "not_null": True,
            },
            False,
            (1,),
        ],
        [
            {
                "col_name": "id",
                "col_dtype": "int",
                "pk": True,
                "not_null": True,
            },
            True,
            None,
        ],
        [
            {
                "col_name": "id",
                "col_dtype": "int",
                "pk": True,
                "not_null": False,
            },
            False,
            (1,),
        ],
        [
            {
                "col_name": "bar",
                "col_dtype": "lorem",
                "pk": False,
                "not_null": False,
            },
            False,
            (1,),
        ],
    ),
)
def test_insert_table(conn, col_dict, dry_run, expected):
    table_name = "foo"
    col = create_column_dict(**col_dict)
    create_table(table_name, [col], dry_run, conn=conn, close_conn=False)

    col_name = col.get("name")
    # returns 1 if a table with the given name exists
    query = f"SELECT 1 FROM PRAGMA_TABLE_INFO('{table_name}') WHERE name='{col_name}';"
    cur = conn.cursor()
    cur.execute(query)
    res = cur.fetchone()
    assert res == expected
    return


# to test:
#  wrong column names
@pytest.mark.parametrize(
    "rows, expect_error, expected_value",
    (
        [
            [
                {
                    "id": 1,
                    "mandatory": "foo",
                    "optional": "bar",
                }
            ],
            False,
            [(1, "foo", "bar")],
        ],  # ok
        [
            [
                {
                    "id": 1,
                    "mandatory": "foo",
                    "optional": None,
                }
            ],
            False,
            [(1, "foo", None)],
        ],  # ok
        [
            [
                {
                    "id": 1,
                    "mandatory": None,
                    "optional": "bar",
                }
            ],
            True,
            None,
        ],  # mandatory missing
        [
            [
                {
                    "id": 1,
                    "mandatory": "foo",
                }
            ],
            True,
            None,
        ],  # too few values
        # [[{"id": None, "mandatory": "foo", "optional": "bar"}], True, None], # this test case requires more work
        [
            [{"id": "hello", "mandatory": "foo", "optional": "bar"}],
            False,
            [("hello", "foo", "bar")],
        ],  # wrong datatype are allowed
        [
            [{"id": 1, "mandatory": 123, "optional": "bar"}],
            False,
            [(1, "123", "bar")],
        ],  # int to text conversion
    ),
)
def test_insert_rows(conn, rows, expect_error, expected_value):
    table_name = "foo"

    # set up table
    id_col = create_column_dict(col_name="id", col_dtype="int", pk=True)
    mandatory_col = create_column_dict(
        col_name="mandatory", col_dtype="text", not_null=True
    )
    optional_col = create_column_dict(col_name="optional", col_dtype="text")
    create_table(
        table_name,
        columns=[id_col, mandatory_col, optional_col],
        conn=conn,
        close_conn=False,
    )

    # insert rows
    if not expect_error:
        insert_rows(table_name, rows=rows, conn=conn, close_conn=False)
        cursor = conn.execute(f"SELECT * FROM {table_name}")
        res = cursor.fetchall()
        assert res == expected_value
    else:
        expected_exceptions = (OperationalError, IntegrityError)
        with pytest.raises(Exception):
            insert_rows(table_name, rows=rows, conn=conn, close_conn=False)
    return
