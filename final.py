
#Import the required libraries
import json
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr
from langchain_community.llms import Ollama

# Step 1: Load and split text files
txt_files = ["fraud_handling_policy.txt", "kyc_policy.txt", "loan_processing_policy.txt", "refund_dispute_policy.txt"]
txt_docs = []

for f in txt_files:
    loader = TextLoader(rf"D:\VS_CODE\INTEL-AIML\bank_nlp\files\{f}", encoding="utf-8")
    txt_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_txt_docs = splitter.split_documents(txt_docs)


# Step 2: Load JSON file
with open(r"D:\VS_CODE\INTEL-AIML\bank_nlp\files\04_qa_pairs.json", "r", encoding="utf-8") as f:

    raw_json = json.load(f)

# Step 3: Convert JSON entries into Document objects (ignoring id and policy_ref)
json_docs = []
for entry in raw_json:
    content = f"Q: {entry['question']}\nA: {entry['answer']}"
    metadata = {
        "category": entry.get("category"),
        "risk_level": entry.get("risk_level"),
        "suggested_action": entry.get("suggested_action"),
        "source": "json"
    }
    json_docs.append(Document(page_content=content, metadata=metadata))


# Step 4: Combine text docs and JSON docs
combined_docs = split_txt_docs + json_docs


# Step 5: Convert to embeddings
emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device':'cpu'})
vect=FAISS.from_documents(combined_docs,emb)
print(f"Total combined docs: {len(combined_docs)}")
vect.save_local("vector_emb")


#Perform similarity search

#Perform similarity search on FAISS vectorstore.
#Returns a list of Document objects (with page_content + metadata).
    
def vector_search(query, top_val=3):
    
    results = vect.similarity_search(query=query, k=top_val)
    return results



import gradio as gr
import json
from langchain_community.llms import Ollama

# Initialize Ollama LLM
llm = Ollama(model="llama3")


import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Load encoder + tokenizer
encoder = AutoModel.from_pretrained(r"D:\VS_CODE\INTEL-AIML\bank_nlp\encoder_model_final_1")
tokenizer = AutoTokenizer.from_pretrained(r"D:\VS_CODE\INTEL-AIML\bank_nlp\tokenizer_model_final_1")

# Recreate heads with same architecture
hidden_size = encoder.config.hidden_size
intent_head = nn.Linear(hidden_size, 4)      # 4 intents
sentiment_head = nn.Linear(hidden_size, 6)   # 6 sentiments

# Load trained weights
intent_head.load_state_dict(torch.load(r"D:\VS_CODE\INTEL-AIML\bank_nlp\intent_head_final.pt", map_location="cpu"))
sentiment_head.load_state_dict(torch.load(r"D:\VS_CODE\INTEL-AIML\bank_nlp\sentiment_head_final.pt", map_location="cpu"))

# Put in eval mode
encoder.eval()
intent_head.eval()
sentiment_head.eval()

# Define mappings (should match training)
intent_map = {"Fraud/Unauthorized": 0, "Loan": 1, "KYC": 2, "Account Access": 3}
sentiment_map = {"Anxious": 0, "Confused": 1, "Urgent": 2, "Neutral": 3, "Frustrated": 4, "Angry": 5}

# Reverse mapping for readability
intent_inv_map = {v: k for k, v in intent_map.items()}
sentiment_inv_map = {v: k for k, v in sentiment_map.items()}


# --- Helper function to format output ---
def format_output(parsed: dict) -> str:
    lines = []
    for k, v in parsed.items():
        key = k.replace("_", " ").capitalize()
        lines.append(f"{key}: {v}")
    return "\n".join(lines)

# --- Intent classification (NLP Part) ---

def clean_text(text):
    import re
    # 1. Lowercase
    text = text.lower()
    
    # 2. Replace ₹ with 'rs'
    text = text.replace("₹", "rs")
    
    # 3. Normalize dates like 25-apr-2023 → <DATE>
    text = re.sub(r"\d{1,2}-[a-z]{3}-\d{4}", "<DATE>", text)
    
    # 4. Normalize numbers → <NUM>
    text = re.sub(r"\d+", "<NUM>", text)
    
    # 5. Remove special characters (keep <DATE> and <NUM>)
    text = re.sub(r"[^a-z0-9\s<DATE><NUM>]", "", text)
    
    # 6. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text.strip()


def classify_intent(query: str):
    
    # Preprocess input
    test_text = clean_text(query)
    enc = tokenizer(test_text,
                padding="max_length",
                truncation=True,
                max_length=128,   # same as training
                return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = encoder(**enc)
        cls_repr = outputs.last_hidden_state[:, 0]
        pred_intent = torch.argmax(intent_head(cls_repr), dim=1)
        pred_sentiment = torch.argmax(sentiment_head(cls_repr), dim=1)

    # Return predictions
    return intent_inv_map[pred_intent.item()], sentiment_inv_map[pred_sentiment.item()]




# --- Chatbot answer (RAG Part) ---
def get_answer(query: str):
    # Step 1: Retrieve relevant docs (assuming vector_search is defined elsewhere)
    docs = vector_search(query, top_val=3)
    context = "\n".join([f"- {d.page_content} | {d.metadata}" for d in docs])

    # Step 2: Build prompt
    prompt = (
        "You are a banking assistant. Respond in JSON format. "
        "Include keys: answer, category, risk_level, suggested_action. "
        "If the query is general or conversational (like greetings or farewells), "
        "return only the 'answer' key."
        "\n\n"
        f"User query: {query}\n\n"
        f"Relevant context:\n{context}\n\n"
        "Generate the JSON response without any extra text."
    )

    try:
        output_text = llm.invoke(prompt).strip()
        if output_text.startswith("```"):
            output_text = output_text.split("```")[1]
        output_text = output_text.replace("json", "").strip()

        try:
            parsed = json.loads(output_text)
        except Exception:
            parsed = {"answer": output_text}
    except Exception as e:
        parsed = {"answer": f"Ollama invoke failed: {e}"}

    return format_output(parsed)

# --- Chat Interface for chatbot ---
def chat_interface(user_message, history):
    if user_message.lower().strip() in ["bye", "exit", "quit"]:
        return "Answer: Goodbye! Have a Nice Day."
    return get_answer(user_message)

# --- Gradio UI ---
with gr.Blocks() as demo:
    # Intro section with styled text and image
    gr.Markdown(
        """
        <div style="text-align:center; font-size:28px; color:#2E86C1; font-weight:bold;">
            🏦 Banking Support & Fraud Intelligence System
        </div>
        <div style="text-align:center; font-size:18px; color:#555;">
            Automating customer support, detecting fraud, and providing context-aware responses
        </div>
        """
    )

    gr.Image(
        value="website_image.png",   # local file in same folder
        type="filepath",
        label="Banking System Overview",
        width=200
    )

    gr.Markdown(
        """
        <div style="font-size:16px; color:#1B4F72; text-align:justify;">
            This project demonstrates an <b>AI-powered banking assistant</b> built with local LLMs (via Ollama) and Retrieval-Augmented Generation (RAG).
            It helps customers with queries related to fraud, loans, KYC, and account access, while also detecting suspicious activities like OTP misuse or phishing attempts.
            Explore the two modules below: <b>NLP Part</b> for intent classification and <b>Chatbot Part</b> for full query answering.
        </div>
        """
    )

    # Tabs for NLP and Chatbot
    with gr.Tab("🔹 NLP Part - Intent and Sentiment Classification"):
      intent_input = gr.Textbox(label="Enter your query")
      intent_output = gr.Textbox(label="Detected Intent")
      sentiment_output = gr.Textbox(label="Detected Sentiment")

      intent_btn = gr.Button("Classify Intent and Sentiment")

      intent_btn.click(
        fn=classify_intent, 
        inputs=intent_input, 
        outputs=[intent_output, sentiment_output]   # <-- two outputs
                      )


    with gr.Tab("🤖 Chatbot - Answer Queries"):
        gr.ChatInterface(
            fn=chat_interface,
            title="💬 Fraud Chat Assistant",
            description="Ask me about fraud investigations, OTP issues, or suspicious transactions."
        )

demo.launch(inbrowser=True)
