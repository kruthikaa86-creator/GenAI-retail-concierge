# GenAI Retail Concierge — RAG + Agentic AI

An end-to-end **Generative AI and Agentic AI retail concierge system** built using OpenAI, LangGraph, ChromaDB, Sentence Transformers, and Streamlit.

The system combines **Retrieval-Augmented Generation (RAG), multi-agent orchestration, customer personalization, tool execution, human-in-the-loop escalation, and agent observability** to simulate a premium retail concierge experience.

## Project Overview

Premium retail brands require intelligent concierge systems that can answer policy questions, discover products, analyze customer feedback, personalize recommendations, and support return requests.

This project implements an AI concierge capable of:

- Grounded product, review, and policy retrieval
- Intent-based agent routing
- Customer preference analysis
- Return status lookup
- SLA risk evaluation
- Human-in-the-loop escalation
- Agent execution tracing
- Interactive Streamlit dashboard and chat interface

## Architecture

```text
User Query
    |
    v
Intent Router
    |
    v
LangGraph Orchestration
    |
    +-------------------------------+
    |              |                |
    v              v                v
RAG Agent     Customer Agent    Return Agent
    |              |                |
    +--------------+----------------+
                   |
                   v
               SLA Check
                   |
             Conditional Edge
               /        \
              v          v
             END    Human Escalation
```

## RAG Pipeline

The RAG knowledge base is built from:

- Product catalog
- Product reviews
- Operations policies

The preprocessing pipeline created:

```text
Product Documents : 77,548
Review Documents  : 155,991
Policy Documents  : 1,050
Total Documents   : 234,589
```

Embeddings are generated using:

`sentence-transformers/all-MiniLM-L6-v2`

The **384-dimensional embeddings** are stored in ChromaDB for semantic retrieval.

Metadata filtering ensures policy, product, and review queries retrieve from the correct knowledge source.

## Agentic Workflow

LangGraph orchestrates the concierge workflow through the following nodes:

- Intent Router
- RAG Agent
- Customer Agent
- Return Agent
- General Agent
- SLA Check
- Human Escalation

The intent router supports six request types:

`policy` · `product` · `review` · `customer` · `return` · `general`

Example workflow:

```text
Recommend products based on my preferences

intent_router
-> customer_agent
-> sla_check
-> human_escalation
```

## Customer Personalization

Customer recommendations use:

- Customer profile
- Customer segment
- Wishlist activity
- Purchase history
- Recent interaction events

The customer agent combines profile and behavioral signals to generate preference-based concierge suggestions.

## Human-in-the-Loop Escalation

The system conditionally escalates high-touch requests.

Escalation conditions include:

- High SLA risk
- Suspended Platinum customer accounts
- Existing escalated return records

Escalated requests are routed for priority human concierge follow-up.

## Observability

Every concierge execution generates an audit trace containing:

- Trace ID
- User query
- Detected intent
- Agent execution path
- Retrieved sources
- Tool outputs
- SLA metadata
- Escalation status
- Final response

This provides end-to-end visibility into agent execution.

## Evaluation Results

The system was evaluated on **20 concierge journeys** covering policy, product discovery, reviews, personalization, returns, and general conversations.

| Metric | Score |
|---|---:|
| Task Completion Rate | 100% |
| Intent Accuracy | 100% |
| Grounded Answer Rate | 100% |
| Escalation Accuracy | 100% |
| Valid Response Rate | 100% |
| Failed Journeys | 0 |

The evaluation pipeline exports scored journey results and summary metrics.

## Streamlit Application

The application includes:

**Dashboard** — Customer, return, SLA, catalog, and review metrics.

**AI Concierge** — Interactive chat connected to the LangGraph workflow.

**Knowledge Base** — Vector-store and indexed document statistics.

**Escalation Queue** — High-touch requests requiring human follow-up.

**Audit Logs** — Recent agent execution traces.

## Tech Stack

- Python 3.11
- OpenAI GPT-4.1 mini
- LangGraph
- ChromaDB
- Sentence Transformers
- Hugging Face
- Pandas
- Streamlit

## Setup

Clone the repository and create a Python 3.11 virtual environment.

```bash
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file:

```text
OPENAI_API_KEY=your_openai_api_key
```

Add the required synthetic retail datasets to the `data` directory and build the vector store.

Run the application:

```bash
streamlit run .\frontend\streamlit.py
```

## Dataset Note

The project uses large synthetic retail datasets for educational and portfolio purposes.

Raw datasets and generated vector-store files are excluded from the public GitHub repository because of file-size constraints.

The final RAG knowledge base contains **234,589 indexed documents**.

## Limitations and Future Improvements

Future improvements include hybrid retrieval, reranking, durable LangGraph checkpointing, enterprise API integration, authentication, LangSmith/OpenTelemetry tracing, and larger adversarial evaluation suites.

## Author

**Kruthika A**

GenAI / Data Science Portfolio Project
