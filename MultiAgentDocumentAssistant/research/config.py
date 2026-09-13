import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    postgres_uri: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    qdrant_url: str
    qdrant_key: str | None = None

    @classmethod
    def from_env(cls):
        return cls(os.environ['POSTGRES_URI'], os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                   os.getenv('NEO4J_USER', 'neo4j'), os.environ['NEO4J_PASSWORD'],
                   os.getenv('QDRANT_URL', 'http://localhost:6333'), os.getenv('QDRANT_API_KEY'))
