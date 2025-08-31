import os
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb

# Ensure directory exists
os.makedirs("tmp", exist_ok=True)

memory = Memory(db=SqliteMemoryDb(table_name="memories", db_file="tmp/agent.db"))
storage = SqliteStorage(table_name="sessions", db_file="tmp/agent.db")

# Optional: explicitly create storage tables
storage.create()