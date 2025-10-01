from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from psycopg2 import sql
from pydantic import Field


class CustomSQLRetriever(BaseRetriever):
    connection: object = Field(...)  # psycopg2 connection, no strict type
    embedding_model: object = Field(...)
    table_name: str = Field(...)
    embedding_column: str = Field(...)
    text_column: str = Field(...)
    k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True  # Allow non-pydantic types like psycopg2 connection

    def get_relevant_documents(self, query):
        # Embed the query
        query_embedding = self.embedding_model.embed_query(query)

        # Use psycopg2.sql for safe identifier interpolation
        sql_query = sql.SQL("""
                            SELECT {text_col}
                            FROM {table}
                            ORDER BY {embedding_col} <#> %s::vector
                                LIMIT %s;
                            """).format(
            table=sql.Identifier(self.table_name),
            embedding_col=sql.Identifier(self.embedding_column),
            text_col=sql.Identifier(self.text_column)
        )

        with self.connection.cursor() as cur:
            cur.execute(sql_query, (query_embedding, self.k))
            rows = cur.fetchall()

        docs = [Document(page_content=row[0]) for row in rows]
        return docs
