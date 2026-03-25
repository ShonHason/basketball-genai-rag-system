# 🏀 AI-Powered NBA League RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system designed to simulate, structure, and query a custom NBA-style league using Large Language Models (LLMs) and vector-based retrieval.

This project focuses on building a realistic AI-driven knowledge system and exploring advanced GenAI techniques such as query rewriting, reranking, and model selection strategies.

---

## 📌 Overview

The system generates a structured sports league (teams, players, contracts, championships) using LLMs, transforms it into a vectorized knowledge base, and enables natural language querying through a multi-stage RAG pipeline.

Beyond basic RAG, the project emphasizes **improving retrieval quality and response accuracy** through iterative experimentation.

---

## 🚀 Key Features

- 🧠 AI-generated structured league data (teams, players, contracts, etc.)
- 📄 Markdown-based knowledge representation
- 🔍 Semantic search using vector embeddings
- 🔄 Retrieval-Augmented Generation (RAG)
- ✨ Query rewriting for improved semantic matching
- 📊 Reranking of retrieved documents
- 🧪 Evaluation using structured JSONL datasets
- ⚙️ Multi-model usage and selection (Grok + OpenAI + OpenRouter)

---

## 🏗️ System Architecture

### 1. Data Generation

- Leveraged multiple LLMs (Grok and OpenAI's GPT-4o-mini) to generate structured league content
- Designed prompts to produce consistent and structured outputs (teams, players, contracts)
- Stored generated data as Markdown documents

---

### 2. Data Processing

- Split documents into semantic chunks
- Converted each chunk into vector embeddings using:
  - `text-embedding-3-large`

---

### 3. Vector Database

- Stored embeddings in **Chroma**
- Enabled semantic similarity search over generated league data

---

### 4. Retrieval Pipeline

- Preserved the original user query
- Generated an additional optimized query (query rewriting)
- Performed retrieval using both queries to improve recall

---

### 5. Reranking Layer

- Retrieved chunks were reranked based on:
  - Relevance to the user query
  - Contextual importance
- Reduced noise from vector similarity search

---

### 6. Response Generation

- Selected top-ranked context
- Passed context + query to LLM (`gpt-4o-mini`)
- Generated final natural language response

---

## 🔄 RAG Flow

1. User submits a query  
2. Query rewriting generates an improved version  
3. Both queries are used for retrieval  
4. Relevant chunks are retrieved from Chroma  
5. Results are reranked  
6. Top context is selected  
7. LLM generates final response  

---

## 🧪 Evaluation & Iteration

- Used JSONL datasets to test system behavior across different queries
- Compared response quality under different:
  - Prompt strategies
  - Retrieval approaches
- Iteratively improved:
  - Prompt design
  - Query rewriting
  - Reranking effectiveness

---

## 🧠 Model Selection Strategy

- Used multiple LLMs (Grok, GPT-4o-mini) based on task requirements
- Evaluated models based on:
  - Output structure
  - Consistency
  - Relevance

### Selection Logic:
- **Data Generation** → Models optimized for creative structured output  
- **Answer Generation** → Models with stronger reasoning and response quality  

---

## 🛠️ Tech Stack

- **LLMs**: OpenAI (GPT-4o-mini), Grok  
- **Embeddings**: text-embedding-3-large  
- **Vector Database**: Chroma  
- **Language**: Python  
- **Data Formats**: Markdown, JSONL  

---

## 🎯 Key Learnings

- Query rewriting significantly improves retrieval performance  
- Combining multiple queries increases recall and answer quality  
- Reranking is critical to reduce noise from vector search  
- Prompt engineering directly impacts system reliability  
- Model selection should be task-specific, not one-size-fits-all  

---

## 🚧 Limitations

- Limited historical depth in generated league data  
- Evaluation is semi-manual (no automated scoring yet)  
- No persistent user personalization  

---

## 🔮 Future Improvements

- Expand dataset with richer historical narratives and stats  
- Introduce automated evaluation metrics  
- Add personalization layer  
- Deploy as a web-based application  

---

## 💡 Why This Project Matters

This project demonstrates practical, hands-on experience in:

- Building end-to-end RAG systems  
- Working with LLM APIs in real-world scenarios  
- Improving GenAI systems through experimentation  
- Applying AI to structured domain problems  

---

## 📬 Contact

Feel free to explore, fork, or reach out!