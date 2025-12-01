import os
import operator
import re
import json
import numpy as np
import nashpy as nash
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

# LangChain / Google Gemini
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)

# Pydantic & Tools
from pydantic import BaseModel, Field
from tavily import TavilyClient
from ddgs import DDGS
from langgraph.graph import StateGraph, END

# Vector Store (FAISS)
from langchain_community.vectorstores import FAISS

# --- LOAD SECRETS ---
load_dotenv()

# --- CONFIGURATION ---
# Slightly higher temp for better argumentation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
CURRENCY = "₹"

# Initialize Tavily
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

MEMORY_DIR = "memory_store"
LOG_PATH = "negotiation_log.jsonl"


# --- LONG-TERM MEMORY (FAISS) ---

class NegotiationMemory:
    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self._load()

    def _load(self):
        if os.path.isdir(MEMORY_DIR):
            try:
                self.vectorstore = FAISS.load_local(
                    MEMORY_DIR,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                self.vectorstore = None

    def _save(self):
        if self.vectorstore is not None:
            try:
                self.vectorstore.save_local(MEMORY_DIR)
            except Exception:
                pass

    def add(self, text: str, meta: dict):
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts([text], embeddings, metadatas=[meta])
        else:
            self.vectorstore.add_texts([text], metadatas=[meta])
        self._save()

    def search(self, query: str, k: int = 3):
        if self.vectorstore is None:
            return []
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception:
            return []


memory = NegotiationMemory()


# --- STATE MANAGEMENT ---

class ProcurementState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    item_name: str
    market_price: int        # Total Market Value (INR)
    max_budget: int          # Absolute Max (Walk-away)
    target_price: int        # Aggressive Target
    my_last_offer: int       # Our last confirmed offer
    seller_last_price: int   # Seller's last stated price
    rounds: int              # Turn counter
    deal_status: str         # 'active', 'closed', 'walked_away', 'pending_human'
    quantity: int            # Unit count
    probe_rounds: int        # Counter for non-price moves
    final_offer_made: bool   # Have we played the "Final Offer" card?
    final_offer_value: int   # The numeric value of that final offer
    pending_decision: Optional[dict]   # For Human-in-the-loop triggers


# --- DATA MODELS ---

class MarketReport(BaseModel):
    estimated_market_value: int = Field(
        description="Average market price found (integer INR for the WHOLE requirement)."
    )
    summary: str = Field(
        description="Brief summary of market conditions (Indian context)."
    )


class SellerUnderstanding(BaseModel):
    seller_price: Optional[int] = Field(
        description="Numeric price the seller is asking for the ENTIRE deal (total INR). Null if no price."
    )
    is_firm: bool = Field(
        description="True if seller implies their price is final/non-negotiable."
    )
    intent: str = Field(
        description="One of: 'negotiate' (countering), 'provide_info' (specs), 'threaten_walk' (pressure), 'terminate_deal' (bye/quit)."
    )
    willingness_to_move: str = Field(
        description="One of: 'high','medium','low','unknown'."
    )
    emotional_tone: str = Field(
        description="One of: 'eager', 'frustrated', 'confident', 'defensive', 'neutral'."
    )


class DialogueResponse(BaseModel):
    response_text: str = Field(
        description="Professional, confident negotiation reply (chat/email style) in English, using INR."
    )
    tactic_used: str = Field(
        description="Brief tag of tactic used (e.g., 'anchor', 'probe', 'save_deal')."
    )
    persuasion_angle: str = Field(
        description="The main argument used: e.g., 'volume_promise', 'market_comp', 'budget_constraint'."
    )


# --- HELPERS: Smart Rounding & Extraction ---

PRICE_REGEX = re.compile(r"(\d[\d,\.]*)")


def smart_round(value: int) -> int:
    """
    Rounds numbers to look like human offers.
    e.g., 146187 -> 146000 or 146500.
    """
    if value < 1000: return value
    if value < 10000:
        return int(round(value / 100) * 100) # Round to nearest 100
    if value < 100000:
        return int(round(value / 500) * 500) # Round to nearest 500
    if value < 1000000: # Less than 10 Lakhs
        return int(round(value / 1000) * 1000) # Round to nearest 1000
    
    # Above 10 Lakhs, round to nearest 10,000 for clean "Lakh" figures
    return int(round(value / 10000) * 10000)


def extract_quantity_from_item(item_name: str) -> int:
    m = re.search(r"(\d+)", item_name)
    if not m:
        return 1
    try:
        return int(m.group(1))
    except ValueError:
        return 1


def fallback_extract_price(text: str) -> Optional[int]:
    clean = text.replace(",", " ")
    m = PRICE_REGEX.search(clean)
    if not m:
        return None
    try:
        return int(float(m.group(1)))
    except ValueError:
        return None


def is_per_unit_phrase(msg: str) -> bool:
    msg = msg.lower()
    keywords = [
        "per unit", "per piece", "per pc", "per head", "per license",
        "per user", "per seat", "per item", "each", "each unit"
    ]
    if any(k in msg for k in keywords):
        return True
    return False


def format_inr(amount: int) -> str:
    # Use standard comma formatting for readability
    return f"{CURRENCY}{amount:,}"


# --- SEARCH TOOL (Smart Retail Anchor) ---

def search_market_price(item_name: str, quantity: int) -> str:
    """
    1. Search for RETAIL price anchors first (Amazon, Flipkart, etc.).
    2. Then search for bulk/corporate pricing.
    """
    # Spec check hack: If "M5" is in name, assume user means "Latest Top Spec" (M3/M4 Max)
    # This prevents the search from finding $0 results for non-existent items.
    query_item = item_name
    if "m5" in item_name.lower():
        query_item = item_name.replace("M5", "M3 Max") # Proxy for future tech
        print(f"   [System] Redirecting search query from 'M5' to 'M3 Max' for realistic pricing...")

    query = f"price of {query_item} in India buy online retail price"
    
    print(f"   ...pinging Tavily for: '{query}'")
    try:
        resp = tavily_client.search(
            query=query,
            search_depth="advanced",
            topic="general",
            max_results=6,
            include_answer=True,
        )
        snippets = []

        answer = resp.get("answer")
        if answer:
            snippets.append(f"Answer: {answer}")

        for r in resp.get("results", []):
            title = r.get("title", "")
            content = r.get("content", "")[:350]
            snippets.append(f"- {title}: {content}")

        if not snippets:
            raise ValueError("No Tavily results")

        return "\n".join(snippets)
    except Exception as e:
        print(f"   Tavily error: {e}, falling back to ddgs")
        try:
            results = list(DDGS().text(query, max_results=5))
            if not results:
                return "No data."
            return "\n".join([f"- {r.get('body', '')}" for r in results])
        except Exception as e2:
            return f"Search error: {e2}"


# --- PARSE SELLER MESSAGE (Intent Detection) ---

def parse_seller_message(state: ProcurementState, message: str) -> SellerUnderstanding:
    parser = llm.with_structured_output(SellerUnderstanding)
    
    prompt = f"""
    You are interpreting a seller's message in a B2B negotiation.

    Message from seller: "{message}"
    Previous seller price: {state.get("seller_last_price", 0) or "none"}.
    Quantity: {state.get("quantity", 1)}.

    Task:
    1. Extract numeric price (Total INR). If per-unit is implied, multiply by Quantity.
    2. Detect INTENT (Crucial):
       - 'negotiate': Counter-offering or arguing value.
       - 'provide_info': Answering questions about specs/delivery.
       - 'threaten_walk': Pressure tactics ("Take it or leave it", "Last offer").
       - 'terminate_deal': Ending the chat ("Bye", "No deal", "I walk", "Done").
    3. Assess Firmness & Tone.

    Return JSON only.
    """
    info = parser.invoke(prompt)

    if info.seller_price is None:
        p = fallback_extract_price(message)
        if p is not None:
            info.seller_price = p

    qty = state.get("quantity", 1)
    if info.seller_price is not None and qty > 1 and is_per_unit_phrase(message):
        # Heuristic to avoid double multiplying
        raw_nums = []
        for m in re.findall(r"\d[\d,]*", message):
            try:
                raw_nums.append(int(m.replace(",", "")))
            except ValueError:
                pass
        
        if not raw_nums:
            info.seller_price = info.seller_price * qty
        else:
            max_num = max(raw_nums)
            diff_ratio = abs(info.seller_price - max_num) / (max_num + 1)
            if diff_ratio > 0.2: 
                 info.seller_price = info.seller_price * qty

    # Apply smart rounding to the extracted price for internal cleanliness
    if info.seller_price:
        info.seller_price = int(info.seller_price)

    return info


# --- MARKET ANALYST (Smart Logic) ---

def estimate_market_value_inr(item_name: str, quantity: int) -> int:
    qty = max(quantity, 1)

    snippets = search_market_price(item_name, qty)

    structured_llm = llm.with_structured_output(MarketReport)
    
    prompt = f"""
    You are a Senior Procurement Analyst in INDIA.

    Item: {item_name}
    Qty: {qty}
    Search Snippets:
    {snippets}

    CRITICAL REALITY CHECK:
    - Look for specific model numbers (e.g., "JBL Live 770NC").
    - If you see a retail price (Amazon/Flipkart), use that as the baseline.
    - Premium items (Apple/MacBook) hold value. 
      - Example: MacBook Pro M3 Max is approx ₹3,00,000 per unit.
      - 50 units = ₹1.5 Crores (1,50,00,000).
    - For bulk (50+ units), apply a ~10-15% discount off retail.
    
    Calculations:
    1. Identify the single unit RETAIL price.
    2. Multiply by {qty}.
    3. Apply the bulk discount factor.

    Return the TOTAL estimated value for ALL {qty} units combined in INR.
    """
    report = structured_llm.invoke(prompt)
    mv = report.estimated_market_value

    if mv < 1000:
        mv = 1000 * qty if qty < 10 else 10000

    return smart_round(mv)


def market_analyst_node(state: ProcurementState):
    item = state["item_name"]
    qty = extract_quantity_from_item(item)
    state["quantity"] = qty

    print(f"[Analyst] Assessing 2025 Indian market rates for: {item} (quantity = {qty}) ...")

    mv_total = estimate_market_value_inr(item, qty)

    # Strategy:
    # Premium products (Apple, JBL) -> Target 80% MV, Max 95% MV
    # Generic products -> Target 60% MV, Max 85% MV
    
    is_premium = any(x in item.lower() for x in ["apple", "mac", "jbl", "bose", "sony", "samsung", "dell", "hp"])
    
    if is_premium:
        target_pct = 0.85 # Less discount for Apple
        max_pct = 0.98 # Willing to pay almost market
    else:
        target_pct = 0.60
        max_pct = 0.85

    target = smart_round(int(mv_total * target_pct))
    max_bud = smart_round(int(mv_total * max_pct))

    print("[Strategy] Estimated Market Value:", format_inr(mv_total))
    print(f"          Target ({int(target_pct*100)}%): ", format_inr(target))
    print(f"          Max ({int(max_pct*100)}%):       ", format_inr(max_bud))

    return {
        "market_price": mv_total,
        "target_price": target,
        "max_budget": max_bud,
        "my_last_offer": 0,
        "seller_last_price": 0,
        "rounds": 0,
        "probe_rounds": 0,
        "deal_status": "active",
        "quantity": qty,
        "final_offer_made": False,
        "final_offer_value": 0,
    }


# --- NASH STANCE DECISION ---

def decide_stance_with_nash(state: ProcurementState, seller_price: Optional[int]) -> str:
    if seller_price is None:
        return "SOFT" # Default to SOFT to encourage a price drop

    buyer_payoffs = np.array([
        [0.2, 0.5, 0.9],  # HARD
        [0.3, 0.7, 0.8],  # MEDIUM
        [0.4, 0.8, 0.7],  # SOFT
    ])
    seller_payoffs = 1.0 - buyer_payoffs

    game = nash.Game(buyer_payoffs, seller_payoffs)
    equilibria = list(game.support_enumeration())

    if not equilibria:
        return "MEDIUM"

    buyer_mix, _ = equilibria[0]
    moves = ["HARD", "MEDIUM", "SOFT"]
    stance = moves[int(buyer_mix.argmax())]

    mv = state.get("market_price", 0) or 1
    ratio = seller_price / mv

    # If seller is very expensive, be HARD
    if ratio >= 1.3:
        return "HARD"
    # If seller is reasonable, we can be SOFT to close
    if ratio <= 1.1:
        return "SOFT"
        
    return "SOFT" # Default to SOFT as per Gym Simulation


# --- NEGOTIATION ENGINE (IQ 200 Logic) ---

def compute_next_offer(
    state: ProcurementState,
    seller_price: Optional[int],
    is_firm: bool,
    willingness: str,
    tone: str,
    intent: str,
) -> tuple[str, int, bool, str]:
    
    T = state["target_price"]
    M = state["max_budget"]
    last = state.get("my_last_offer", 0)
    rounds = state.get("rounds", 0)
    final_made = state.get("final_offer_made", False)
    final_val = state.get("final_offer_value", 0)
    probe_count = state.get("probe_rounds", 0)

    # --- 0. HANDLE TERMINATION (Smart Save) ---
    if intent == "terminate_deal":
        # IQ 200: If they walk, but price is <= Max, SAVE THE DEAL.
        if seller_price and seller_price <= M:
             return "accept_deal", seller_price, True, "save_deal_at_walkaway"
        
        # If slightly above Max (within 5%), try one last desperate bridge
        if seller_price and seller_price <= M * 1.05:
            return "final_offer", M, False, "desperate_save_attempt"
            
        return "walk_away", last, False, "seller_terminated"

    if intent == "threaten_walk" or is_firm:
        # If they threaten walk and we are close, DO NOT lowball. Bridge the gap.
        if seller_price and seller_price <= M:
            return "accept_deal", seller_price, True, "threat_accepted"
        
        return "final_offer", M, False, "threat_response_max"


    # --- 1. FINAL OFFER MODE ---
    if final_made:
        if seller_price is None:
            return "final_offer", final_val, False, "restate_final"

        S = seller_price
        
        # Win condition
        if S <= final_val:
            return "accept_deal", S, True, "seller_accepted_final"

        # Manager Approval Close (The "Smart Close")
        if S <= M:
            return "accept_deal", S, True, "manager_approval_close"

        # Above Max -> Walk
        return "walk_away", final_val, False, "above_max_after_final"


    # --- 2. STANDARD MODE ---
    if seller_price is None:
        # Don't probe twice in a row. If we probed last time, anchor now.
        if probe_count > 0:
             return "make_offer", T, False, "blind_anchor"
        return "probe", last if last > 0 else 0, False, "no_price_probe"

    S = seller_price

    # Insane price check
    if S > M * 3:
        if rounds == 0: return "probe", last, False, "insane_price_probe"
        return "walk_away", last, False, "insane_price_walk"

    # Immediate Win
    if S <= T:
        return "accept_deal", S, True, "below_target_win"

    # Nash Stance
    stance = decide_stance_with_nash(state, S)
    if stance == "HARD":
        concession_rate = 0.10 # Move 10% of gap
    elif stance == "SOFT":
        concession_rate = 0.40 # Move 40% of gap
    else:
        concession_rate = 0.25 # Move 25% of gap

    # RECIPROCITY LOGIC (The "Specter" fix)
    # If seller moved down significant amount, we MUST move up.
    seller_prev = state.get("seller_last_price", 0)
    seller_moved = False
    if seller_prev > 0 and S < seller_prev:
        seller_moved = True
        concession_rate += 0.15 

    # Strategic Final Offer (Endgame)
    if rounds >= 5:
        # If we are close (within 10%), just close it at Max or Split
        if S <= M * 1.1:
             return "final_offer", min(S, M), False, "time_limit_close"

    # Counter-Offer Calculation
    if last == 0: 
        # First offer: Anchor at Target
        new_offer = T
    else:
        # Subsequent offers: Bridge the gap based on concession rate
        gap = S - last
        if gap < 0: gap = 0
        step = int(gap * concession_rate)
        
        # Ensure minimum movement if seller moved (don't be stubborn)
        if seller_moved and step < (M - T)*0.05:
            step = int((M - T)*0.05) 
            
        new_offer = last + step

    new_offer = smart_round(new_offer)
    if new_offer > M: new_offer = M
    
    # Trivial Gap Close
    if S > 0 and abs(S - new_offer)/S < 0.03:
        return "accept_deal", S, True, "gap_trivial_accept"

    return "make_offer", new_offer, False, f"counter_{stance.lower()}"


# --- NEGOTIATOR NODE (Mark's Persona) ---

def negotiator_node(state: ProcurementState):
    messages = state["messages"]
    last_human_msg = messages[-1].content

    # 1. Parse Input
    seller_info = parse_seller_message(state, last_human_msg)

    # 2. Compute Strategy
    decision_label, offer_price, closed, reason_tag = compute_next_offer(
        state,
        seller_price=seller_info.seller_price,
        is_firm=seller_info.is_firm,
        willingness=seller_info.willingness_to_move,
        tone=seller_info.emotional_tone,
        intent=seller_info.intent,
    )

    # 3. Guardrails & Rounding
    current_offer = state.get("my_last_offer", 0)
    
    # Never regress (go lower than before) unless it's a specific tactic
    if offer_price < current_offer and current_offer > 0:
        offer_price = current_offer
        
    offer_price = smart_round(offer_price)
    
    # 4. Prompt Generation
    offer_str = format_inr(offer_price)
    
    # Dynamic Tactic Guidance
    tactic_guidance = ""
    
    if decision_label == "make_offer":
        tactic_guidance = (
            f"Counter with {offer_str}. Do NOT ask for a breakdown. "
            "Give a strong COMMERCIAL ARGUMENT (e.g., 'For 50 units, this is the market clearing price', "
            "'We can process the PO today', 'This aligns with our Q4 budget'). "
            "Be persuasive, not inquisitive."
        )
    elif decision_label == "probe":
        tactic_guidance = (
            "Do not give a number yet. "
            "State that you need to understand the spec compliance or warranty terms before committing to a price. "
            "Put the pressure back on them to justify their premium."
        )
    elif decision_label == "final_offer":
        tactic_guidance = (
            f"State clearly: '{offer_str} is our ceiling.' "
            "Do not say 'Best and Final' typically, say 'This is the limit of my authorization.' "
            "Make it sound like a policy constraint, not a choice."
        )
    elif decision_label == "accept_deal":
        if reason_tag == "save_deal_at_walkaway":
             tactic_guidance = (
                 "The seller is walking. STOP THEM. "
                 f"Say 'Wait. I made a call. We can do {offer_str}. Let's sign now.' "
                 "Be urgent."
             )
        elif reason_tag == "manager_approval_close":
            tactic_guidance = (
                f"Agree to {offer_str}. Say you pushed this through with special approval "
                "because you value the relationship. Close the deal."
            )
        else:
            tactic_guidance = "Accept the deal professionally. Confirm next steps (PO issuance)."
    elif decision_label == "walk_away":
        tactic_guidance = (
            "State that the numbers simply do not work. "
            "Wish them well professionally and end the call. "
            "Do not ask questions. Be decisive."
        )

    # 5. Context Retrieval
    similar_cases = memory.search(last_human_msg, k=2)
    cases_text = "\n".join(doc.page_content for doc in similar_cases) if similar_cases else "None."

    dialogue_prompt = f"""
    You are Mark, Senior Procurement Manager at Northbridge Systems.
    Persona: You are a "Harvey Specter" type closer. High IQ, low tolerance for inefficiency.
    You do not ask begging questions like "Can you give me a breakdown?".
    You give ARGUMENTS. You trade LEVERAGE.
    
    Context:
    Item: {state['item_name']} (Qty: {state['quantity']})
    Seller said: "{last_human_msg}"
    
    Decision: {decision_label}
    My Number: {offer_str}
    Reason: {reason_tag}
    
    INSTRUCTIONS:
    - Use clean numbers (e.g., 1,45,000 not 1,45,231).
    - NO EMOJIS.
    - Be concise (max 3 sentences).
    - If countering, give a reason (Volume, Cash Terms, Future Potential).
    - {tactic_guidance}
    
    Output JSON.
    """
    structured_dial = llm.with_structured_output(DialogueResponse)
    response = structured_dial.invoke(dialogue_prompt)

    # 6. Update State
    new_status = state.get("deal_status", "active")
    new_probe_r = state.get("probe_rounds", 0)
    pending_decision = state.get("pending_decision", None)

    if decision_label == "probe":
        new_probe_r += 1

    if decision_label == "walk_away":
        new_status = "pending_human"
        pending_decision = {"action": "walk_away", "price": offer_price}
    
    if closed:
        new_status = "pending_human"
        pending_decision = {"action": "closed", "price": offer_price}

    rounds = state.get("rounds", 0) + 1

    # 7. Persist Memory & Logs
    log_entry = (
        f"Round {rounds}: Seller said '{last_human_msg}'. "
        f"Mark replied '{response.response_text}'. Offer: {offer_str}"
    )
    memory.add(log_entry, {"item": state['item_name']})

    try:
        log_record = {
            "item": state["item_name"],
            "round": rounds,
            "seller_msg": last_human_msg,
            "mark_reply": response.response_text,
            "decision": decision_label,
            "offer": offer_price
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")
    except:
        pass

    return {
        "messages": [AIMessage(content=response.response_text)],
        "my_last_offer": offer_price,
        "seller_last_price": seller_info.seller_price or state.get("seller_last_price", 0),
        "deal_status": new_status,
        "probe_rounds": new_probe_r,
        "rounds": rounds,
        "final_offer_made": state.get("final_offer_made", False),
        "final_offer_value": state.get("final_offer_value", 0),
        "pending_decision": pending_decision,
    }


# --- GRAPH & UI HOOKS ---

def build_procurement_graph():
    graph = StateGraph(ProcurementState)
    graph.add_node("analyst", market_analyst_node)
    graph.set_entry_point("analyst")
    graph.add_edge("analyst", END)
    return graph.compile()

def run_ai_turn(state: ProcurementState, seller_input: str):
    state["messages"].append(HumanMessage(content=seller_input))
    result = negotiator_node(state)
    
    ai_msg = result["messages"][0].content
    for k, v in result.items():
        if k != "messages":
            state[k] = v
            
    state["messages"].append(AIMessage(content=ai_msg))
    return state, ai_msg


# --- MAIN (CLI) ---

def main():
    print("--- MARK: THE BUYER (IQ 200) ---")
    item_name = input("Enter item to procure: ").strip()

    state: ProcurementState = {
        "messages": [],
        "item_name": item_name,
        "market_price": 0, "max_budget": 0, "target_price": 0,
        "my_last_offer": 0, "seller_last_price": 0,
        "rounds": 0, "probe_rounds": 0, "deal_status": "active",
        "quantity": 1, "final_offer_made": False, "final_offer_value": 0,
        "pending_decision": None,
    }

    graph = build_procurement_graph()
    state = graph.invoke(state)

    print(f"\n[Analyst] MV: {format_inr(state['market_price'])} | Max: {format_inr(state['max_budget'])}")

    intro = f"This is Mark, Senior Procurement Manager at Northbridge Systems."
    opening = "We have cleared the budget for this requirement. Please share your best all-inclusive price in INR."
    mark_opening = intro + " " + opening
    
    print(f"\nMark: {mark_opening}")
    state["messages"].append(AIMessage(content=mark_opening))

    while True:
        if state["deal_status"] in ["closed", "walked_away", "pending_human"]:
            break
        
        seller_in = input("\nSeller: ").strip()
        if seller_in.lower() in ["quit", "exit"]: break
        
        state, reply = run_ai_turn(state, seller_in)
        print(f"\nMark: {reply}")

    if state["deal_status"] == "pending_human":
        action = state["pending_decision"]["action"]
        price = state["pending_decision"]["price"]
        print(f"\n[SYSTEM] Mark wants to {action.upper()} at {format_inr(price)}. (Approve in UI)")

if __name__ == "__main__":
    main()