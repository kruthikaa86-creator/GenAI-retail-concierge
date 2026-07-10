# GenAI Retail Concierge — RAG + Agentic Automation

An end-to-end Generative AI and Agentic AI retail concierge system built using OpenAI, Retrieval-Augmented Generation (RAG), LangGraph, ChromaDB, Sentence Transformers, and Streamlit.

The system retrieves grounded retail knowledge, understands customer intent, uses customer and return tools, evaluates SLA risk, conditionally escalates high-touch requests to human agents, and records complete execution traces for observability.

---

## Project Overview

Premium retail brands provide concierge services such as personalized product assistance, return support, policy guidance, and high-touch customer service.

Scaling this experience digitally requires more than a traditional chatbot.

This project implements an AI-powered retail concierge capable of:

- Retrieving product, review, and operations policy information
- Generating grounded responses using RAG
- Routing customer requests based on detected intent
- Using customer profile and behavioral interaction data
- Providing preference-based product suggestions
- Looking up return records and return status
- Evaluating operational SLA risk
- Escalating high-touch customer requests
- Logging complete agent execution traces
- Providing a Streamlit chat interface and operations dashboard

---

## System Architecture

```text
User Query
    |
    v
Intent Router
    |
    v
LangGraph Agent Orchestration
    |
    +-----------------------------+
    |              |              |
    v              v              v
RAG Agent     Customer Agent   Return Agent
    |              |              |
    +--------------+--------------+
                   |
                   v
              SLA Check
                   |
            Conditional Edge
              /          \
             v            v
           END     Human Escalation
                            |
                            v
                           END
```

The final response and agent execution information are recorded in the observability layer and displayed through the Streamlit application.

---

## Agent Routes

The intent router classifies customer queries into six intents.

| Intent | Purpose |
|---|---|
| `policy` | Return eligibility, refund rules, and operational policies |
| `product` | Product discovery, category search, stock, brand, and pricing |
| `review` | Product reviews, ratings, and customer feedback |
| `customer` | Customer preferences, wishlist, purchase history, and personalization |
| `return` | Return lookup, return status, and return resolution |
| `general` | Greetings and general concierge interactions |

### Example

```text
Query:
Recommend products based on my preferences

Intent:
customer

Agent Path:
intent_router
-> customer_agent
-> sla_check
-> human_escalation
```

---

## Retrieval-Augmented Generation

The RAG knowledge base is built using:

- `products.csv`
- `product_reviews.csv`
- `operations_policies.json`

The preprocessing pipeline converts the source records into retrieval-ready documents with structured metadata.

### Vector Store Statistics

```text
Product Documents: 77,548
Review Documents: 155,991
Policy Documents: 1,050

Total RAG Documents: 234,589
```

Embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

ChromaDB is used as the persistent vector database.

Metadata filtering restricts retrieval to the relevant knowledge source.

Examples:

```text
Policy Query
-> Retrieve policy documents only

Product Query
-> Retrieve product documents only

Review Query
-> Retrieve product review documents only
```

Product-specific review queries also use Product ID filtering.

---

## OpenAI Generation Layer

The project uses the OpenAI API with:

```text
gpt-4.1-mini
```

The model is used for:

- Intent classification
- Grounded concierge response generation

Generation guardrails enforce:

- Responses based only on supplied context
- No invented policy decisions
- No invented product details
- Exact source price preservation
- Exact currency preservation
- No unsupported currency conversion
- Review summaries based only on supplied ratings and review text
- English responses unless another language is explicitly requested
- Clarification questions when required policy conditions are missing
- No exposure of internal RAG or agent architecture to customers

---

## LangGraph Agentic Workflow

LangGraph is used to orchestrate the multi-agent concierge workflow.

Implemented nodes include:

```text
Intent Router
RAG Agent
Customer Agent
Return Agent
General Agent
SLA Check
Human Escalation
```

Conditional edges dynamically route requests based on intent and escalation requirements.

### RAG Workflow

```text
Intent Router
-> RAG Agent
-> SLA Check
-> END
```

### Personalized Customer Workflow

```text
Intent Router
-> Customer Agent
-> SLA Check
-> Human Escalation
```

### Return Workflow

```text
Intent Router
-> Return Agent
-> SLA Check
-> Human Escalation
```

---

## Concierge Tools

Local CSV-backed tools simulate enterprise operational systems.

Implemented tools include:

- Customer profile lookup
- Customer interaction lookup
- Customer preference analysis
- Product lookup by Product ID
- Customer return history lookup
- Return lookup by Return ID
- SLA risk evaluation

The project intentionally simulates operational tools using local datasets and does not require live enterprise APIs.

---

## Customer Personalization

The customer agent combines:

```text
customers.csv
+
customer_interactions.csv
+
products.csv
```

Customer signals include:

- Customer segment
- Lifetime value
- Loyalty points
- Account status
- Wishlist activity
- Purchase activity
- Recent interaction events

The system uses these signals to generate preference-based concierge suggestions.

Wishlist events represent recorded customer interest.

Purchase events represent recorded purchase history.

The system does not assume customer satisfaction unless supported by the available data.

---

## Return Support

The return agent supports:

- Return lookup by Return ID
- Customer return history lookup
- Return status reporting
- Return reason reporting
- Resolution type reporting
- Refund amount reporting

Example:

```text
Query:
What is the status of return R0000001?

Intent:
return

Agent Path:
intent_router
-> return_agent
-> sla_check
-> human_escalation
```

---

## Human-in-the-Loop Escalation

The concierge system evaluates whether human intervention is required.

Implemented escalation conditions include:

- High SLA risk
- Suspended Platinum customer accounts
- Existing escalated return records

Example:

```text
Escalation Required: True

Reason:
Platinum customer account is suspended
```

Escalated requests receive priority concierge handoff messaging.

---

## SLA Risk Evaluation

The system evaluates operational SLA metrics using:

```text
agent_performance_sla.csv
```

Example SLA output:

```text
Risk Level: medium
Average SLA Met: 84.69%
Average First Response Time: 15.31 minutes
Total Escalations: 3818
```

SLA risk is evaluated before completing the agent workflow.

---

## Observability and Agent Tracing

Every concierge execution is recorded as a JSONL trace.

Trace records contain:

- Trace ID
- Timestamp
- User query
- Customer ID
- Return ID
- Detected intent
- Agent execution path
- Retrieved source metadata
- Tool outputs
- SLA metadata
- Escalation status
- Escalation reason
- Final response

Example agent path:

```text
intent_router
-> customer_agent
-> sla_check
-> human_escalation
```

Trace logs are stored locally under the observability logs directory.

Generated trace logs are excluded from GitHub.

---

## Streamlit Application

The project includes an interactive Streamlit application.

### Dashboard

Displays:

- Total customers
- Platinum customer count
- Pending returns
- SLA compliance
- Return status distribution
- Customer segment distribution
- Product catalog count
- Product review count
- Escalated returns

### AI Concierge

Provides an interactive chat interface connected to the LangGraph concierge workflow.

The interface also displays:

- Agent execution path
- Detected intent
- Escalation status
- Escalation reason
- SLA risk metadata
- Retrieved source metadata

### Knowledge Base

Displays:

- Vector store document count
- Product document count
- Product review document count
- Policy document count
- Embedding model
- Vector database information

### Escalation Queue

Displays requests requiring human concierge follow-up.

### Audit Logs

Displays recent agent execution traces.

---

## Dataset Scale

| Dataset | Records |
|---|---:|
| Agent Performance SLA | 334,329 |
| Chat Conversation Metadata | 128,475 |
| Customers | 93,551 |
| Customer Interactions | 161,532 valid loaded rows |
| Products | 77,548 |
| Product Reviews | 155,991 |
| Returns | 152,285 |
| Operations Policies | 1,050 |

The `customer_interactions.csv` source contains malformed rows.

The ingestion pipeline detects the parsing issue and loads valid records while skipping malformed records.

Large raw datasets are excluded from the GitHub repository because of repository size constraints.

---

## Evaluation

The system was evaluated using 20 concierge journeys covering:

- Policy questions
- Product discovery
- Product reviews
- Customer personalization
- Return inquiries
- General concierge conversations

### Final Evaluation Results

| Metric | Score |
|---|---:|
| Task Completion Rate | 100% |
| Intent Accuracy | 100% |
| Grounded Answer Rate | 100% |
| Escalation Accuracy | 100% |
| Valid Response Rate | 100% |
| Failed Journeys | 0 |

The evaluation pipeline initially identified an ambiguous product recommendation query that was incorrectly routed to the customer agent.

The intent routing rules were refined to distinguish general product recommendations from explicit customer-personalized recommendations.

The complete 20-journey evaluation was then rerun successfully.

Evaluation outputs include:

```text
evaluation/outputs/journey_evaluation.csv
evaluation/outputs/evaluation_summary.json
```

---

## Project Structure

```text
genai-retail-concierge/
|
|-- agents/
|   |-- concierge_graph.py
|   `-- intent_router.py
|
|-- dashboard/
|   `-- dashboard_service.py
|
|-- data/
|   `-- README.md
|
|-- embeddings/
|   |-- embedding_model.py
|   `-- vector_store.py
|
|-- evaluation/
|   |-- evaluate_journeys.py
|   `-- outputs/
|
|-- exploration/
|   `-- data_exploration.py
|
|-- frontend/
|   `-- streamlit.py
|
|-- generation/
|   `-- generator.py
|
|-- ingestion/
|   `-- data_loader.py
|
|-- observability/
|   `-- trace_logger.py
|
|-- preprocessing/
|   |-- document_creator.py
|   `-- inspect_rag_data.py
|
|-- rag/
|   `-- rag_pipeline.py
|
|-- retrieval/
|   `-- retriever.py
|
|-- tools/
|   |-- concierge_tools.py
|   `-- inspect_tool_data.py
|
|-- .gitignore
|-- README.md
`-- requirements.txt
```

---

## Installation

Python 3.11 is recommended.

### 1. Clone the repository

```bash
git clone <repository-url>
cd genai-retail-concierge
```

### 2. Create a virtual environment

```bash
py -3.11 -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure the OpenAI API key

Create a `.env` file in the project root.

```text
OPENAI_API_KEY=your_openai_api_key
```

Never commit the `.env` file to GitHub.

### 6. Add the datasets

Place the required synthetic retail datasets inside the `data` directory.

Required files:

```text
agent_performance_sla.csv
chat_conversation_metadata.csv
customers.csv
customer_interactions.csv
operations_policies.json
products.csv
product_reviews.csv
returns.csv
```

The raw datasets are not included in this repository because of file-size constraints.

### 7. Build the vector store

```bash
python .\embeddings\vector_store.py
```

### 8. Run the Streamlit application

```bash
streamlit run .\frontend\streamlit.py
```

### 9. Run the evaluation pipeline

```bash
python .\evaluation\evaluate_journeys.py
```

---

## Technology Stack

- Python 3.11
- OpenAI API
- GPT-4.1 mini
- LangGraph
- ChromaDB
- Sentence Transformers
- Hugging Face
- Pandas
- Streamlit

---

## Limitations

- All datasets are synthetic and intended for education and portfolio demonstrations.
- Operational tools simulate enterprise systems using local files.
- Raw datasets are excluded from the public repository because of file-size constraints.
- The vector store must be generated locally.
- Customer interaction data contains malformed source records that are skipped during loading.
- Retrieval quality depends on the supplied synthetic catalog and policy content.
- Evaluation currently uses a fixed 20-journey test suite.
- Authentication and role-based access control are not implemented.
- State-changing enterprise actions are simulated.

---

## Production Next Steps

Future production improvements include:

- Enterprise database and API integration
- Authentication and role-based access control
- Hybrid semantic and lexical retrieval
- Cross-encoder reranking
- Durable LangGraph checkpointing
- Human approval workflows for state-changing actions
- LangSmith or OpenTelemetry tracing
- Prompt and model version tracking
- Automated regression evaluation
- Adversarial RAG testing
- CI/CD evaluation gates

---

## Author

**Kruthika A**

GenAI / Data Science Portfolio Project
