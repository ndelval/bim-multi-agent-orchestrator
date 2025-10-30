# Quick Start: Testing Memory Persistence Fix

## ⚡ Fast Test (2 minutes)

```bash
# Start chat with hybrid memory
python -m orchestrator.cli chat --memory-provider hybrid

# Test conversation memory
You: Me llamo Nicolas
# Wait for response...

You: Sabes como me llamo?
# System should now remember your name! ✅

You: exit
```

## 📊 What to Look For

### ✅ Success Indicators

**In the logs:**
```
INFO - Conversation turn stored in memory: <uuid>
INFO - Hybrid vector: 3 hits  (instead of 0 hits)
INFO - Hybrid lexical: 2 hits
```

**In the response:**
- Second query: System says "Sí, te llamas Nicolas" or similar
- NOT: "No, no sé cómo te llamas" ❌

### ❌ Failure Indicators

**In the logs:**
```
INFO - Hybrid vector: 0 hits  (still 0!)
WARNING - Failed to store conversation in memory
```

**In the response:**
- System still doesn't remember name
- No storage logs appear

## 🔍 Debug Commands

### Check if anything was stored:
```python
import chromadb
client = chromadb.PersistentClient(path=".orchestrator/hybrid_chroma")
collection = client.get_collection("hybrid_memory")
print(f"Documents stored: {collection.count()}")
```

### Check SQLite lexical storage:
```python
import sqlite3
conn = sqlite3.connect(".orchestrator/hybrid_lexical.db")
cursor = conn.cursor()
count = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
print(f"Lexical entries: {count}")
```

### View stored conversations:
```python
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.core.config import MemoryConfig, MemoryProvider

config = MemoryConfig(provider=MemoryProvider.HYBRID)
manager = MemoryManager(config)

# Search for conversations
results = manager.retrieve("Nicolas", limit=5)
for result in results:
    print(f"Content: {result.get('content')}")
    print(f"Metadata: {result.get('metadata')}")
    print("---")
```

## 🐛 Common Issues

### Issue 1: "No module named 'rich'"
```bash
pip install rich
# or
uv pip install rich
```

### Issue 2: "Memory provider not initialized"
```bash
# Check that Neo4j is running (for graph storage)
docker ps | grep neo4j

# Or start it:
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### Issue 3: "ChromaDB directory not found"
```bash
# Create the directory manually
mkdir -p .orchestrator/hybrid_chroma

# Then restart chat
python -m orchestrator.cli chat --memory-provider hybrid
```

## 📝 Advanced Test: Multi-Turn Conversation

```bash
python -m orchestrator.cli chat --memory-provider hybrid

You: Me llamo Nicolas y vivo en España
# System acknowledges

You: ¿Cuál es mi nombre?
# Should say: "Tu nombre es Nicolas" ✅

You: ¿Dónde vivo?
# Should say: "Vives en España" ✅

You: ¿Qué sabes sobre mí?
# Should recall both name and location ✅
```

## 🎯 Expected Performance

- **Storage latency**: < 100ms per turn
- **Retrieval latency**: < 500ms per query
- **Memory accuracy**: 100% for recent conversations
- **Database growth**: ~3 entries per conversation turn
  - 1 entry: user message
  - 1 entry: assistant message
  - 1 entry: combined context

## 🚀 Next Steps After Successful Test

1. **Add user session management** - Replace "default_user" with actual user IDs
2. **Configure memory TTL** - Prevent unbounded growth
3. **Add conversation boundaries** - Separate different conversation sessions
4. **Enable graph queries** - Test Neo4j relationship queries

## 📞 Need Help?

If tests fail, check:
1. `MEMORY_FIX_VALIDATION.md` - Full implementation details
2. Logs at: `orchestrator/cli/chat_orchestrator.py:434` (storage logs)
3. Memory provider logs: `orchestrator/memory/providers/hybrid_provider.py`

**Key log lines to grep:**
```bash
grep "Conversation turn stored" <logfile>
grep "Hybrid vector:" <logfile>
grep "Failed to store" <logfile>
```
