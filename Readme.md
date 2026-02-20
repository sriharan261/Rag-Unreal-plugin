Here’s a clean, well-structured Markdown version with improved hierarchy, formatting, and readability:

---

# 🦙 LlamaRAG Unreal Engine Plugin

**LlamaRAG** is a native, offline Retrieval-Augmented Generation (RAG) plugin for Unreal Engine 5.
Powered by **llama.cpp**, it allows you to run Large Language Models (LLMs) and vector embeddings natively in **C++** and **Blueprints**.

With this plugin, your NPCs or in-game systems can read `.txt` files containing lore, rules, or character backstories and answer player questions dynamically — **without an internet connection, Python servers, or API costs**.

---

# 🛠️ 1. Installation & Integration

## Step 1 — Copy the Plugin Folder

1. Navigate to your Unreal project root directory (where the `.uproject` file exists).
2. Create a folder named `Plugins` if it doesn’t exist.
3. Place the **LlamaRAG** plugin folder inside:

```
YourProject/Plugins/LlamaRAG/
```

---

## Step 2 — Ensure Third-Party Binaries Are Present

The plugin depends on compiled **llama.cpp** binaries.

Expected folder structure:

```
YourProject/Plugins/LlamaRAG/ThirdParty/LlamaCpp/
├── include/   # llama.h, ggml.h, etc.
├── lib/       # llama.lib
└── bin/       # llama.dll, ggml.dll, ggml-base.dll, etc.
```

> ⚠️ The `bin` folder **must** contain all required DLLs.
> The plugin loads them explicitly to prevent initialization crashes.

---

## Step 3 — Regenerate and Build

1. Right-click `YourProject.uproject`
2. Select **Generate Visual Studio project files**
3. Open the `.sln` file
4. Build using:

```
Configuration: Development Editor
Platform: Win64
```

5. Launch Unreal Editor
6. Go to **Edit → Plugins** and confirm **Llama Local RAG** is enabled.

---

# 🧠 2. Setting Up AI Models & Data

The plugin requires:

* 1️⃣ LLM model (`.gguf`)
* 2️⃣ Embedding model (`.gguf`)
* 3️⃣ Your text data

## Create Required Folders

Inside your project `Content` directory:

```
Content/
├── Models/
└── Data/
```

---

## Download an LLM

Download a quantized `.gguf` model from HuggingFace, for example:

* `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`
* `Phi-3-mini-4k-instruct.gguf`

Place inside:

```
Content/Models/
```

---

## Download an Embedding Model

Example:

```
nomic-embed-text-v1.5.f16.gguf
```

Place inside:

```
Content/Models/
```

---

## Create Your Lore File

Create:

```
Content/Data/story.txt
```

Add your:

* Game lore
* NPC backstory
* Rules
* World knowledge

---

# 📘 3. Blueprint Node Reference

All functionality is exposed via the **Llama RAG Component**.

Add this component to:

* Actor
* GameMode
* PlayerController
* NPC Controller

---

## 🧩 Helper Nodes

### **Get Content Directory Path** *(Pure Node)*

**Description:**
Returns the absolute path to the project `Content` folder.
Works in both Editor and Packaged builds.

---

## ⚡ Action Nodes (Async)

### **Load Models Async**

Loads llama.cpp and initializes LLM + embedding models in a background thread.

**Inputs**

* `LLMModelPath` (String)
* `EmbedModelPath` (String)

---

### **Ingest Story From File Async**

Reads a `.txt` file, chunks text, generates embeddings, and stores them in RAM.

**Inputs**

* `FilePath` (String)
* `ChunkSize` (Integer, Default: `500`)

---

### **Ask Question Async**

Searches ingested text and generates contextual answers.

**Inputs**

* `Question` (String)
* `TopK` (Integer, Default: `3`)
  Determines number of retrieved context chunks.

---

## 🔔 Event Dispatchers (Callbacks)

AI operations run in background threads. Bind to these events:

### **On Models Loaded**

* Fires when models finish loading
* Output: `bSuccess` (Boolean)

---

### **On Story Ingested**

* Fires when text embedding completes
* Output: `bSuccess` (Boolean)

---

### **On Answer Generated**

* Fires when LLM finishes generating
* Output: `Answer` (String)

---

# 🎮 4. Detailed Blueprint Usage Guide

Example Blueprint: `BP_NPC_Controller`

---

## Step 1 — Add Component

1. Open Blueprint
2. Click **Add Component**
3. Select **Llama RAG Component**

---

## Step 2 — Load Models (BeginPlay)

From `Event BeginPlay`:

1. Call **Load Models Async**
2. Build paths:

```
Get Content Directory Path
      ↓
Append String
      ↓
Models/your_llm_name.gguf
```

Repeat for embedding model.

Add event:

```
On Models Loaded
```

---

## Step 3 — Ingest Lore File

From `On Models Loaded`:

1. Add Branch → check `bSuccess == true`
2. Call:

```
Ingest Story From File Async
```

Path example:

```
Get Content Directory Path → Append → Data/story.txt
```

Bind:

```
On Story Ingested
```

When triggered:

👉 Enable player chat UI — AI is ready.

---

## Step 4 — Ask Questions

When player submits a question:

```
Ask Question Async
```

Bind:

```
On Answer Generated
```

Use returned `Answer` to:

* Display in Dialogue UI
* Print to screen
* Send to Text-To-Speech

---

✅ **Your offline RAG NPC system is now fully integrated into Unreal Engine!**

---