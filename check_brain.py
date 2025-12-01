import json
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

LOG_PATH = "negotiation_log.jsonl"
MEMORY_DIR = "memory_store"

def check_brain():
    print("--- 🧠 MARK'S BRAIN SCAN ---")
    
    # 1. CHECK LOGIC ADAPTATION (JSONL)
    if not os.path.exists(LOG_PATH):
        print("❌ No negotiation logs found yet. Negotiate with Mark first!")
        return

    print(f"\n[1] STRATEGIC ADAPTATION (from {LOG_PATH})")
    stats = {"HARD": 0, "MEDIUM": 0, "SOFT": 0}
    wins = {"HARD": 0, "MEDIUM": 0, "SOFT": 0}
    
    with open(LOG_PATH, "r") as f:
        for line in f:
            data = json.loads(line)
            # Infer stance from reason tag (heuristic)
            reason = data.get("reason", "")
            stance = "MEDIUM"
            if "pressure" in reason or "firm" in reason: stance = "HARD"
            elif "accept" in reason or "save" in reason: stance = "SOFT"
            
            stats[stance] += 1
            if data.get("decision") == "accept_deal":
                wins[stance] += 1

    for stance in ["HARD", "MEDIUM", "SOFT"]:
        total = stats[stance]
        win_count = wins[stance]
        rate = (win_count/total * 100) if total > 0 else 0
        print(f"   - {stance} Stance: Used {total} times | Win Rate: {rate:.1f}%")

    # 2. CHECK LONG-TERM MEMORY (FAISS)
    print(f"\n[2] EPISODIC MEMORY (from {MEMORY_DIR})")
    if os.path.exists(MEMORY_DIR):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vectorstore = FAISS.load_local(MEMORY_DIR, embeddings, allow_dangerous_deserialization=True)
            count = vectorstore.index.ntotal
            print(f"   ✅ Memory Active. Mark remembers {count} distinct negotiation turns.")
        except Exception as e:
            print(f"   ⚠️ Error reading memory: {e}")
    else:
        print("   ⚠️ No long-term memory formed yet.")

if __name__ == "__main__":
    check_brain()