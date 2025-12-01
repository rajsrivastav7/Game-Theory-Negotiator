import streamlit as st
from langchain_core.messages import AIMessage

from negotiator import (
    build_procurement_graph,
    run_ai_turn,
    format_inr,
    ProcurementState,
)

# ---------- PAGE CONFIG & GLOBAL STYLING ----------

st.set_page_config(
    page_title="Game Theory Negotiator",
    page_icon="💼",
    layout="wide",
)

# Black background, white bold Consolas text, professional look.
# FIX: Removed the global wildcard '*' selector to prevent breaking icons.
# Applied font specifically to text containers and form elements.
st.markdown(
    """
    <style>
    /* Main Background and Text Color */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-family: "Consolas", monospace !important;
    }

    /* Apply Consolas font to headers, inputs, buttons, and custom classes */
    h1, h2, h3, h4, h5, h6, p, span, div {
        font-family: "Consolas", monospace; 
    }
    
    /* Ensure form elements use the custom font */
    input, textarea, button, select, .stTextInput, .stButton {
        font-family: "Consolas", monospace !important;
        font-weight: 700 !important;
    }

    /* Specific exclusion to fix Material Icons (the "keyboard_double_arrow_right" issue) */
    .material-icons, .material-icons-outlined, .material-icons-round, .material-icons-sharp, .material-icons-two-tone {
        font-family: 'Material Icons' !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Chat message containers */
    .chat-message-user {
        background-color: #111111;
        border: 1px solid #333333;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    .chat-message-assistant {
        background-color: #050505;
        border: 1px solid #444444;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    .chat-role {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #AAAAAA;
        margin-bottom: 0.2rem;
        font-weight: 700 !important;
    }

    .chat-text {
        font-size: 0.95rem;
        color: #FFFFFF;
        white-space: pre-wrap;
        font-weight: 700 !important;
    }

    .divider {
        border-top: 1px solid #333333;
        margin: 0.75rem 0 0.75rem 0;
    }

    .system-badge {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888888;
        font-weight: 700 !important;
    }

    .metric-box {
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SESSION STATE INITIALIZATION ----------

if "state" not in st.session_state:
    st.session_state["state"] = {
        "messages": [],
        "item_name": "",
        "market_price": 0,
        "max_budget": 0,
        "target_price": 0,
        "my_last_offer": 0,
        "seller_last_price": 0,
        "rounds": 0,
        "probe_rounds": 0,
        "deal_status": "active",
        "quantity": 1,
        "final_offer_made": False,
        "final_offer_value": 0,
        # for human-in-the-loop
        "pending_decision": None,
    }

if "chat" not in st.session_state:
    # list[dict(role: "user"/"assistant", content: str)]
    st.session_state["chat"] = []

if "initialized" not in st.session_state:
    st.session_state["initialized"] = False

# ---------- SIDEBAR: STATUS / METRICS ----------

with st.sidebar:
    st.markdown("### Game Theory Negotiator")
    st.markdown(
        "<span class='system-badge'>MODE: MARK – BUYER (INR)</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    state: ProcurementState = st.session_state["state"]
    if st.session_state["initialized"]:
        st.markdown("**Current Deal Snapshot**")
        st.markdown(
            f"<div class='metric-box'>"
            f"<b>Item:</b> {state['item_name']}<br>"
            f"<b>Market Value:</b> {format_inr(state['market_price'])}<br>"
            f"<b>Target:</b> {format_inr(state['target_price'])}<br>"
            f"<b>Max Budget:</b> {format_inr(state['max_budget'])}<br>"
            f"<b>Rounds:</b> {state['rounds']}<br>"
            f"<b>Status:</b> {state['deal_status'].upper()}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "Enter an item on the right panel to start a new negotiation."
        )

# ---------- MAIN LAYOUT ----------

st.markdown("## Game Theory Negotiator – Mark (INR, India)")

# If not initialized, show item input form
if not st.session_state["initialized"]:
    st.markdown("### Configure a new negotiation")

    with st.form("init_form", clear_on_submit=False):
        item_name = st.text_input(
            "Item(s) to procure",
            placeholder="e.g. 50 office chairs, 100 laptops, AMC for servers",
        )
        submitted = st.form_submit_button("Start Negotiation")

    if submitted and item_name.strip():
        item_name = item_name.strip()

        # Build graph and run analyst
        graph = build_procurement_graph()
        base_state: ProcurementState = {
            "messages": [],
            "item_name": item_name,
            "market_price": 0,
            "max_budget": 0,
            "target_price": 0,
            "my_last_offer": 0,
            "seller_last_price": 0,
            "rounds": 0,
            "probe_rounds": 0,
            "deal_status": "active",
            "quantity": 1,
            "final_offer_made": False,
            "final_offer_value": 0,
            "pending_decision": None,
        }
        analyzed_state = graph.invoke(base_state)

        # Prepare Mark's intro – NO first price from buyer
        intro = (
            "This is Mark, Senior Procurement Manager at Northbridge Systems. "
            "I will be leading this negotiation on behalf of our firm."
        )
        opening = (
            f"For {analyzed_state['item_name']}, we have completed our internal market benchmarking "
            "and budget approvals. To start constructively, please share your best all-inclusive price "
            "in INR for the full requirement."
        )
        # Join with a space so we don't leak literal \n\n into the UI
        mark_opening = intro + " " + opening

        # Update state
        analyzed_state["messages"].append(AIMessage(content=mark_opening))
        # IMPORTANT: Mark has NOT yet made a numeric offer
        analyzed_state["my_last_offer"] = 0


        st.session_state["state"] = analyzed_state
        st.session_state["chat"] = [
            {"role": "assistant", "content": mark_opening},
        ]
        st.session_state["initialized"] = True
        st.rerun()

else:
    # ---------- CHAT HISTORY RENDERING ----------

    st.markdown("### Negotiation Thread")

    for msg in st.session_state["chat"]:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(
                f"""
                <div class="chat-message-user">
                    <div class="chat-role">SELLER</div>
                    <div class="chat-text">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-message-assistant">
                    <div class="chat-role">MARK (BUYER)</div>
                    <div class="chat-text">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ---------- INPUT / HUMAN-IN-LOOP / SUMMARY ----------

    state: ProcurementState = st.session_state["state"]

    # 1) Negotiation finished: show summary
    if state["deal_status"] in ["closed", "walked_away"]:
        if state["deal_status"] == "closed":
            final_price = state["my_last_offer"]
            savings = state["market_price"] - final_price
            if savings > 0:
                pct = savings / state["market_price"] * 100
                summary = (
                    f"Negotiation closed at {format_inr(final_price)}. "
                    f"Surplus captured versus market value: {format_inr(savings)} "
                    f"({pct:.1f}% below MV)."
                )
            else:
                summary = (
                    f"Negotiation closed at {format_inr(final_price)} "
                    f"(at or above estimated market value)."
                )
        else:
            seller_price = state.get("seller_last_price", 0)
            if seller_price:
                summary = (
                    f"Negotiation ended without agreement. "
                    f"Last seller price: {format_inr(seller_price)}. "
                    "Pricing remained above disciplined thresholds."
                )
            else:
                summary = (
                    "Negotiation ended without agreement. "
                    "Seller never reached a value within disciplined thresholds."
                )

        st.markdown(f"**Negotiation Summary**\n\n{summary}")
        st.info(
            "Refresh the page or change the item in the form above to start a new negotiation."
        )

    else:
        # 2) Human-in-the-loop: pending decision
        pending = state.get("pending_decision")

        if pending and state["deal_status"] == "pending_human":
            st.warning(
                f"Mark intends to {pending['action']} at {format_inr(pending['price'])}. "
                "Do you approve this move?"
            )
            c1, c2 = st.columns(2)
            with c1:
                approve = st.button("Approve")
            with c2:
                reject = st.button("Reject")

            if approve:
                # Finalize according to pending action
                if pending["action"] == "closed":
                    state["deal_status"] = "closed"
                else:
                    state["deal_status"] = "walked_away"
                state["pending_decision"] = None
                st.session_state["state"] = state
                st.rerun()
            elif reject:
                # Override: continue the negotiation instead of closing
                state["deal_status"] = "active"
                state["pending_decision"] = None
                st.session_state["state"] = state
                st.rerun()

        else:
            # 3) Active negotiation: seller input box
            seller_msg = st.text_input(
                "Your message as the Seller",
                placeholder="e.g. My best price is 8,00,000 all-inclusive.",
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                send = st.button("Send")

            if send and seller_msg.strip():
                # Append seller message to chat
                st.session_state["chat"].append(
                    {"role": "user", "content": seller_msg.strip()}
                )

                # Run one negotiation turn
                updated_state, mark_reply = run_ai_turn(
                    st.session_state["state"], seller_msg.strip()
                )
                st.session_state["state"] = updated_state
                st.session_state["chat"].append(
                    {"role": "assistant", "content": mark_reply}
                )

                st.rerun()