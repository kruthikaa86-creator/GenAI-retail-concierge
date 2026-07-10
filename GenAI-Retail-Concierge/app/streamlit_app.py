from pathlib import Path
import sys

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.graph.workflow import build_workflow


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Customer Support Chatbot",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}

[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
}

[data-testid="stHeader"] {
    background-color: #0e1117;
}

.main .block-container {
    max-width: 1120px;
    padding-top: 5.2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    padding-bottom: 4rem;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background-color: #262730;
    min-width: 300px;
    max-width: 300px;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #262730;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 5.1rem;
    padding-left: 20px;
    padding-right: 20px;
}

.nav-active {
    display: block;
    width: 100%;
    box-sizing: border-box;
    background-color: #4b4d5c;
    color: #ffffff;
    padding: 5px 9px;
    border-radius: 5px;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 5px;
}

.nav-item {
    display: block;
    width: 100%;
    box-sizing: border-box;
    color: #d6d6df;
    padding: 5px 9px;
    font-size: 15px;
    font-weight: 400;
    margin-bottom: 1px;
}


/* TITLE */

h1 {
    color: #fafafa !important;
    font-size: 42px !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    margin-bottom: 18px !important;
}

h2 {
    color: #fafafa !important;
    font-size: 30px !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    margin-top: 25px !important;
    margin-bottom: 18px !important;
}

.support-subtitle {
    color: #8b8d98;
    font-size: 14px;
    margin-top: -7px;
    margin-bottom: 28px;
}


/* LABELS */

label {
    color: #fafafa !important;
}

[data-testid="stWidgetLabel"] p {
    color: #fafafa !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}


/* SELECT BOX */

div[data-baseweb="select"] > div {
    background-color: #262730 !important;
    border: 1px solid #262730 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    min-height: 43px !important;
}

div[data-baseweb="select"] span {
    color: #ffffff !important;
}


/* TEXT INPUT */

div[data-baseweb="input"] > div {
    background-color: #262730 !important;
    border: 1px solid #262730 !important;
    border-radius: 8px !important;
    min-height: 43px !important;
}

div[data-baseweb="input"] input {
    color: #ffffff !important;
    font-size: 15px !important;
}

div[data-baseweb="input"] input::placeholder {
    color: #a6a7b0 !important;
    opacity: 1 !important;
}


/* BUTTON */

.stButton > button {
    background-color: #ff4b4b !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 7px !important;
    min-height: 41px !important;
    padding-left: 17px !important;
    padding-right: 17px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}

.stButton > button:hover {
    background-color: #ff5b5b !important;
    color: #ffffff !important;
    border: none !important;
}


/* ASSISTANT RESPONSE */

.assistant-response {
    color: #fafafa;
    font-size: 15px;
    line-height: 1.7;
    margin-top: 4px;
    margin-bottom: 24px;
}


/* CHAT HISTORY */

.chat-user {
    background-color: #1b1e26;
    color: #ffffff;
    border-radius: 7px;
    padding: 17px 18px;
    margin-top: 10px;
    margin-bottom: 18px;
    font-size: 15px;
}

.chat-assistant {
    color: #fafafa;
    padding: 4px 10px;
    margin-bottom: 24px;
    font-size: 15px;
    line-height: 1.7;
}

.empty-history {
    color: #8b8d98;
    font-size: 14px;
}

.source-line {
    color: #8b8d98;
    font-size: 12px;
    margin-top: 5px;
    margin-bottom: 20px;
}


/* HIDE STREAMLIT FOOTER AND MENU */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# WORKFLOW
# ============================================================

@st.cache_resource
def load_workflow():
    """
    Load the compiled LangGraph workflow.
    """

    return build_workflow()


def create_initial_state(
    user_query: str,
    customer_id: str,
) -> dict:
    """
    Create the initial LangGraph state.
    """

    return {
        "user_query": user_query,
        "customer_id": customer_id,
        "confidence": 0.0,
        "sla_risk": False,
        "requires_escalation": False,
        "escalation_reason": "",
        "trace": [],
    }


workflow = load_workflow()


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "assistant_response" not in st.session_state:
    st.session_state.assistant_response = ""


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="nav-active">streamlit</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">admin</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">ai</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">customers</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">help</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">integrations</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">knowledge</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">ops</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">security</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "Retail Customer Support Chatbot"
)


st.markdown(
    '<div class="support-subtitle">'
    'Supports inventory, promotions, and general support queries.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# RAG STRATEGY
# ============================================================

rag_strategy = st.selectbox(
    "RAG Strategy",
    options=[
        "agentic_rag",
    ],
)


# ============================================================
# LLM MODEL
# ============================================================

llm_model = st.selectbox(
    "LLM Model",
    options=[
        "openai",
    ],
)


# ============================================================
# QUESTION INPUT
# ============================================================

user_query = st.text_input(
    "Ask your question",
    placeholder=(
        "Example: Is Wireless Earbuds in stock? "
        "Any promotion on electronics?"
    ),
)


# ============================================================
# GET ANSWER
# ============================================================

get_answer = st.button(
    "Get Answer"
)


# ============================================================
# PROCESS REQUEST
# ============================================================

if get_answer:

    if not user_query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        customer_id = "C000002"

        initial_state = create_initial_state(
            user_query=user_query.strip(),
            customer_id=customer_id,
        )

        with st.spinner(
            "Processing your request..."
        ):

            try:

                result = workflow.invoke(
                    initial_state
                )

                final_answer = result.get(
                    "final_answer",
                    "No answer was generated.",
                )

                st.session_state.assistant_response = (
                    final_answer
                )

                st.session_state.chat_history.append(
                    {
                        "question": user_query.strip(),
                        "answer": final_answer,
                        "intent": result.get(
                            "intent",
                            "unknown",
                        ),
                        "sources": result.get(
                            "retrieval_sources",
                            [],
                        ),
                    }
                )

            except Exception as error:

                st.error(
                    "The concierge workflow "
                    "could not complete the request."
                )

                st.exception(
                    error
                )


# ============================================================
# ASSISTANT RESPONSE
# ============================================================

if st.session_state.assistant_response:

    st.markdown(
        "## Assistant Response"
    )

    st.markdown(
        st.session_state.assistant_response
    )


# ============================================================
# CHAT HISTORY
# ============================================================

st.markdown(
    "## Chat History"
)


if not st.session_state.chat_history:

    st.markdown(
        '<div class="empty-history">'
        'No conversations yet.'
        '</div>',
        unsafe_allow_html=True,
    )


else:

    for chat in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            '<div class="chat-user">'
            '🔴 &nbsp;'
            + chat["question"]
            + '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="chat-assistant">'
            '🤖 &nbsp;'
            + chat["answer"]
            + '</div>',
            unsafe_allow_html=True,
        )

        sources = chat.get(
            "sources",
            [],
        )

        if sources:

            st.markdown(
                '<div class="source-line">'
                'Sources: '
                + " ".join(sources)
                + '</div>',
                unsafe_allow_html=True,
            )