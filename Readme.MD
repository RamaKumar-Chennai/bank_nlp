# AI-Powered Banking Support & Fraud Intelligence System (NLP + RAG)

## 📌 Project Overview
Banks face large volumes of customer queries daily (fraud, loans, KYC, account issues). Current responses are manual, slow, and inconsistent.  
This project builds an AI system that automates support, detects fraud, and assists agents with decision-making using NLP + RAG.

## 🎯 Objectives
- Automate customer support responses
- Detect potential fraud cases in real time
- Provide context-aware answers via RAG
- Suggest next actions (block card, escalate, verify)

## 🛠 Skills & Tools
- Python scripting  
- Machine Learning & NLP  
- Retrieval Systems (RAG)  
- Vector Databases (FAISS/ChromaDB)  
- Streamlit for UI  

## 📂 Dataset Sources
1. Customer Support Tickets (queries, categories, resolutions)  
2. Banking Policy Documents (PDF/text for RAG ingestion)  
3. Transaction Dataset (structured fraud labels)  
4. Synthetic Fraud Scenarios  

## ⚙️ System Workflow
1. **NLP Layer** → Intent + Sentiment classification  
2. **Retriever** → Vector DB (FAISS/ChromaDB)  
3. **RAG Pipeline** → Context-aware response generation  
4. **Action Layer** → Suggest next steps  
5. **Gradio Dashboard** → User interface  



