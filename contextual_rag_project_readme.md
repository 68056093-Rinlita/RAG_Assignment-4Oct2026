# Contextual Retrieval-Augmented Generation (Contextual RAG)

**Course Assignment:** KDAI-NLP  
**Student ID:** 68056093  
**Repository:** [RAG_Assignment-4Oct2026](https://github.com/68056093-Rinlita/RAG_Assignment-4Oct2026)

---

## 📌 Overview

โปรเจกต์นี้เป็นการศึกษารูปแบบ **Contextual RAG (Retrieval-Augmented Generation)** ตามแนวทางของ Anthropic โดยประยุกต์ใช้ **OpenSearch** สำหรับการค้นหาแบบ **Hybrid Search (BM25 + Dense Vector Search)** และสังเคราะห์คำตอบด้วยโมเดล **Qwen** ผ่าน OpenRouter API

ในการทดลองนี้ได้มีการเปรียบเทียบประสิทธิภาพระหว่าง **Base RAG System** (ระบบตั้งต้นตามสไลด์) และ **Improved RAG System** (ระบบที่ผ่านการปรับแต่ง Chunking Strategy และ Contextual Prompt) เพื่อประเมินผลลัพธ์ของ Retrieval และ Final Generation

---

## 🏗 System Architecture & Flow

ระบบทำงานร่วมกัน 2 ส่วนหลัก:

1. **Document Ingestion & Context Generation:**
   * อ่านเอกสาร Markdown จากโฟลเดอร์ `corpus_input/` และทำการตัดแบ่ง Chunk
   * ส่ง Chunk + Full Document ไปให้ LLM สรุปบริบทเฉพาะ (Context Generation)
   * นำข้อความและบริบทไปจัดเก็บใน OpenSearch ผ่าน 2 Indices:
     * `anthropic-vector-index`: จัดเก็บ Embedding ของ `Chunk + Context`
     * `anthropic-bm25-index`: จัดเก็บ Text แยกฟิลด์ `content` และ `contextualized_content`

2. **Hybrid Search & Generation:**
   * ทำงานแบบ Hybrid Search ดึงข้อมูลพร้อมกันทั้ง Dense Vector Search และ Sparse BM25 Search
   * รวมคะแนนและจัดลำดับด้วย **Reciprocal Rank Fusion (RRF)**
   * ส่ง Top-K Chunks ที่มีความเกี่ยวข้องสูงให้ LLM สังเคราะห์คำตอบสุดท้าย

---

## 🔄 Comparison: Base vs. Improved System (ข้อ 2 & ข้อ 3)

### 1. ตารางเปรียบเทียบการตั้งค่า (Configuration Comparison)

| หัวข้อการปรับแต่ง | ระบบเดิม (1) Base System | ระบบปรับปรุง (2) Improved System |
| :--- | :--- | :--- |
| **Directory** | `Base/` | `Improved/` |
| **Chunk Size** | `256` ตัวอักษร | `512` ตัวอักษร |
| **Chunk Overlap** | `0` ตัวอักษร | `50` ตัวอักษร |
| **Context Prompt Language** | ภาษาอังกฤษ (English Prompt) | ภาษาไทย (Thai Prompt) |
| **Context Scope** | สรุปบริบททั่วไปเป็นภาษาอังกฤษ | สรุปบริบทภาษาไทยที่ตรงกับคอร์ปัสและภาษาของคำถาม |

---

### 2. วิเคราะห์ผลลัพธ์และเหตุผลความแตกต่าง (Why & How)

#### A. การปรับแต่ง Chunk Size (256 → 512) และใส่ Overlap (50)
* **ปัญหาของระบบเดิม:** ภาษาไทยเป็นภาษาที่เขียนติดกันโดยไม่มีการเว้นวรรคระหว่างคำ การตัด Chunk ขนาดเล็กเกินไป (`256`) และไม่มีส่วนซ้อนทับ (`overlap = 0`) ทำให้ประโยคหรือใจความสำคัญมักโดนตัดครึ่งบริบท
* **การปรับปรุงและผลลัพธ์:** การขยายขนาดเป็น `512` ร่วมกับ `chunk_overlap = 50` ช่วยรักษาความต่อเนื่องของเนื้อหาระหว่างขอบของแต่ละ Chunk ป้องกันการขาดหายของบริบท ทำให้ Dense Vector Search จับความหมายทางอรรถศาสตร์ (Semantic Meaning) ได้สมบูรณ์ยิ่งขึ้น

#### B. การเปลี่ยน Context Generation Prompt เป็นภาษาไทย
* **ปัญหาของระบบเดิม:** เอกสารต้นทาง (Corpus) และคำถามค้นหาของผู้ใช้เป็นภาษาไทยทั้งหมด แต่ระบบเดิมสั่งให้ LLM สร้าง Contextual Explanation เป็นภาษาอังกฤษ ทำให้การเก็บข้อมูลคีย์เวิร์ดใน BM25 Sparse Index ไม่ตรงกับภาษาของคำถาม
* **การปรับปรุงและผลลัพธ์:** การปรับ Prompt ให้สังเคราะห์ Context เป็นภาษาไทยตรงกับคอร์ปัส ทำให้ฟิลด์ `contextualized_content` ใน BM25 Index มีคีย์เวิร์ดภาษาไทยที่สอดคล้องกับภาษาที่ผู้ใช้ใช้ค้นหา ส่งผลให้ BM25 Retrieval ทำงานได้แม่นยำขึ้นอย่างชัดเจน

#### C. ผลกระทบต่อ Final Answer
* เมื่อประสิทธิภาพของทั้ง Dense Vector และ Sparse BM25 เพิ่มขึ้น ข้อมูล Chunks ที่ถูกดึงขึ้นมาผ่าน RRF Fusion จึงมีความเกี่ยวข้องสูง (High Relevance) และมีเนื้อหาที่สมบูรณ์ LLM จึงนำข้อมูลไปเรียบเรียงและตอบคำถามสุดท้ายได้อย่างถูกต้อง ครบถ้วน และไม่เกิดการหลูบหรือสร้างข้อมูลเท็จ (Hallucination)

---

## 📂 Repository Structure

```text
RAG_Assignment-4Oct2026/
├── Base/
│   ├── authenticRAG.py                      # สคริปต์สร้าง Vector Index สำหรับระบบ Base
│   ├── onlysearchAuthenticRAG.py             # สคริปต์ ค้นหาและตอบคำถาม สำหรับระบบ Base
│   └── authentic_rag_search_results.json    # ผลลัพธ์การค้นหาและคำตอบของระบบ Base
├── Improved/
│   ├── Improved_authenticRAG.py             # สคริปต์สร้าง Vector Index สำหรับระบบ Improved (Chunk 512 + Thai Prompt)
│   ├── Improved_onlysearchAuthenticRAG.py    # สคริปต์ ค้นหาและตอบคำถาม สำหรับระบบ Improved
│   └── improved_authentic_rag_search_results.json # ผลลัพธ์การค้นหาและคำตอบของระบบ Improved
├── corpus_input/                            # คลังเอกสารต้นทาง (.md)
│   ├── 1.md
│   ├── 2.md
│   ├── 44.md
│   └── 5555.md
└── README.md                                # เอกสารอธิบายโปรเจกต์และการเปรียบเทียบ
```

---

## 🛠 Prerequisites & Installation

### 1. Environment Setup

```bash
conda create -n advrag python=3.10
conda activate advrag

pip install opensearch-py sentence-transformers llama-index llama-index-embeddings-huggingface openai tqdm
pip install langchain==0.1.6 langchain-community==0.0.19 langchain-core==0.1.23
```

### 2. Run OpenSearch Container (Docker)

```bash
docker network create opensearch-net

docker run -d --name opensearch -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "network.host=0.0.0.0" \
  opensearchproject/opensearch:2.19.1
```

### 3. API Key & Local Embedding Setup

```cmd
set OPENROUTER_API_KEY=your_openrouter_api_key
ollama pull bge-m3
```

---

## 🚀 Execution Guide

### Step 1: Running Base Model

```cmd
# เข้าไปที่โฟลเดอร์ Base
cd Base

# 1. นำเข้าเอกสารและสร้าง Vector Index
python authenticRAG.py

# 2. ค้นหาและสร้างคำตอบสรุป (ผลลัพธ์จะเซฟที่ authentic_rag_search_results.json)
python onlysearchAuthenticRAG.py
```

---

### Step 2: Clear OpenSearch Database & Running Improved Model

```cmd
# 1. ลบข้อมูล Index เดิมใน OpenSearch ออกก่อนทำตัว Improved
curl -X POST "localhost:9200/anthropic-*/_delete_by_query?pretty" -H "Content-Type: application/json" -d "{\"query\": {\"match_all\": {}}}"

# 2. เข้าไปที่โฟลเดอร์ Improved
cd ../Improved

# 3. นำเข้าเอกสารและสร้าง Vector Index ด้วยการปรับปรุงใหม่
python Improved_authenticRAG.py

# 4. ค้นหาและสร้างคำตอบสรุป (ผลลัพธ์จะเซฟที่ improved_authentic_rag_search_results.json)
python Improved_onlysearchAuthenticRAG.py
```

---

## ✉️ Submission Info

* **Mail Subject:** `KDAI-NLP-68056093-Assignment-4Oct2026`
* **Student ID:** `68056093`
* **Course:** Natural Language Processing (NLP) / Data Science