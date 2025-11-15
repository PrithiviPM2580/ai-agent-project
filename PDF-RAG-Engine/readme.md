# PDF RAG Engine — Documentation

## Overview

This project demonstrates a minimal retrieval-augmented search (RAG) pipeline using LangChain components in TypeScript. The sample script at `src/pdf-rag-engine/index.ts` performs the following high-level steps:

- Load PDF documents from disk.
- Split the documents into chunks.
- Convert chunks to embeddings using Google's Generative AI embeddings.
- Store embeddings in an in-memory vector store and run similarity search / retrieval.

This README documents how to configure, run, and understand the code in `index.ts`.

## Prerequisites

- Node.js (v18+ recommended; your environment shows Node.js v22).
- `pnpm` as the package manager (project uses `pnpm` by default).
- A Google API key with access to the embeddings model (set in `GOOGLE_API_KEY`).

## Project layout

- `src/pdf-rag-engine/index.ts` — main example script (load, split, embed, index, retrieve).
- `PDF-RAG-Engine/resume_X.pdf` — example PDF file path referenced in the code (replace with your PDF).

## Environment

Create a `.env` file at project root with the following variable:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

The code imports `dotenv` via `import "dotenv/config"` so the `.env` file will be loaded automatically.

## Install dependencies

From the project root run:

```bash
pnpm install
```

Notes:

- If you encounter pnpm virtual store or peer dependency warnings, run `pnpm install` and follow the prompt or set `--strict-peer-dependencies=false` if you understand the tradeoffs.
- If `tsx` complains about a missing `esbuild`, install it as a dev dependency:

```bash
pnpm add -D esbuild
```

## Run the example

To run the script in watch mode (project uses `tsx`):

```bash
pnpm run dev
```

Or run once with `tsx`:

```bash
npx tsx src/pdf-rag-engine/index.ts
```

## Code walkthrough (matching `index.ts`)

1. Documents

- Purpose: Demonstrates creating LangChain `Document` objects directly for small examples.
- Example in code: two `Document` objects are created with `pageContent` and `metadata`.

2. PDF loader

- Module used: `@langchain/community/document_loaders/fs/pdf` (`PDFLoader`).
- Behavior: `new PDFLoader("./resume_X.pdf")` loads the PDF file and returns an array of `Document` objects (one per page or chunk depending on the loader internals).
- Action required: Replace `./resume_X.pdf` with the path to your PDF file.

3. Text Splitters

- Module used: `@langchain/textsplitters` (`RecursiveCharacterTextSplitter`).
- Purpose: Breaks long documents into smaller chunks suitable for embedding and retrieval.
- Config in example: `chunkSize: 500`, `chunkOverlap: 100`.

4. Embeddings

- Module used: `@langchain/google-genai` (`GoogleGenerativeAIEmbeddings`).
- Configuration: `model: "text-embedding-004"`, `apiKey: process.env.GOOGLE_API_KEY`.
- Notes: Ensure your Google API key is correct and has access to the embeddings model you specify.

5. Vector store

- Module used: `@langchain/classic/vectorstores/memory` (`MemoryVectorStore`).
- Purpose: Stores document embeddings in-memory for fast similarity search in examples.
- Production note: For large datasets or persistence, replace with a persistent vector store (e.g., Pinecone, Chroma, Weaviate).

6. Retriever & queries

- The code creates a retriever from the vector store with MMR search (`searchType: "mmr"`) and runs a batch of queries with `retriever.batch([...])`.
- Example queries: `"What is the name of the pdf owner?"`, `"List the project names mentioned in the pdf?"`.

## Expected output

- The script logs the retrieval results for the batch of queries to the console. Each retrieved item includes the matched document content and its metadata.

## Troubleshooting

- Error "Cannot find package 'esbuild' imported from .../tsx/dist/cli.mjs": install esbuild as a dev dependency with `pnpm add -D esbuild`.
- pnpm virtual store mismatch: run `pnpm install` from project root to align the virtual store. If prompted, confirm replacing `node_modules`.
- Peer dependency conflicts during install: update packages to compatible versions or use the pnpm config flag `--strict-peer-dependencies=false` when running `pnpm install` (understand the tradeoffs).

## Next steps / Improvements

- Replace `MemoryVectorStore` with a scalable/persistent vector DB for larger corpora.
- Add caching and batching for embeddings to reduce API calls and costs.
- Integrate an LLM for generating answers (RAG) using the retrieved passages.

## References

- LangChain TypeScript docs: https://js.langchain.com/
- Google Generative AI embeddings docs (your Google console)

---
