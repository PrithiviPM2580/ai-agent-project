// ============================================
//  🔹 Index
// ============================================

// ------------------------------------------------------
// 1️⃣ Document
// ------------------------------------------------------
import { Document } from "@langchain/core/documents";

const document = [
  new Document({
    pageContent:
      "Dogs are great companions, known for their loyalty and friendliness.",
    metadata: { source: "mammal-pets-doc" },
  }),
  new Document({
    pageContent: "Cats are independent pets that often enjoy their own space.",
    metadata: { source: "mammal-pets-doc" },
  }),
];
// console.log(document);
// ------------------------------------------------------
// 2️⃣ Pdf loader
// ------------------------------------------------------
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const loader = new PDFLoader("./resume_X.pdf");
const pdfDocs = await loader.load();
// console.log(pdfDocs[0].pageContent.slice(0, 200));
// console.log(pdfDocs[0].metadata);
// console.log(pdfDocs.length);

// ------------------------------------------------------
// 3️⃣ Text Splitters
// ------------------------------------------------------
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100,
});

const allSplits = await textSplitter.splitDocuments(pdfDocs);
// console.log(allSplits);
// console.log(allSplits.length);

// ------------------------------------------------------
// 4️⃣ Embeddings
// ------------------------------------------------------
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import "dotenv/config";
import { assert } from "console";

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  apiKey: process.env.GOOGLE_API_KEY,
});

const vector1 = await embeddings.embedQuery(allSplits[0].pageContent);
const vector2 = await embeddings.embedQuery(allSplits[1].pageContent);

// assert(vector1.length === vector2.length);
// console.log(`Generated vectors of length ${vector1.length}\n`);
// console.log(vector1);
// console.log(vector2);

// ------------------------------------------------------
// 5️⃣ Vectors stores
// ------------------------------------------------------
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

const vectorStores = new MemoryVectorStore(embeddings);
await vectorStores.addDocuments(allSplits);
const results = await vectorStores.similaritySearch(
  "What is the name of the pdf owner"
);
// console.log(results);

// ------------------------------------------------------
// 6️⃣ Retriveal
// ------------------------------------------------------

const retriever = vectorStores.asRetriever({
  searchType: "mmr",
  searchKwargs: {
    fetchK: 1,
  },
});

const resultsBatch = await retriever.batch([
  "What is the name of the pdf owner?",
  "List the project names mentioned in the pdf?",
]);
console.log(resultsBatch);
