// ============================================
//  🔹 RAG Chatbot
// ============================================

// ------------------------------------------------------
// 1️⃣ Components (Google Gemini)
// ------------------------------------------------------
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import "dotenv/config";

// Initialize the chat model
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  apiKey: process.env.GOOGLE_API_KEY,
});

// Initialize the embeddings model
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  apiKey: process.env.GOOGLE_API_KEY,
});

// ------------------------------------------------------
// 2️⃣ Vector Store (Memory)
// ------------------------------------------------------
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

// Create a vector store
const vectorStore = new MemoryVectorStore(embeddings);

// ------------------------------------------------------
// 3️⃣ Indexing (Load Document, Split and Store)
// ------------------------------------------------------

//Load Documents from a web page using Cheerio
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

const pTageSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTageSelector,
  }
);

const docs = await cheerioLoader.load();

// console.assert(docs.length == 1);
// console.log("Document", docs);
// console.log("Total charaterstics: ", docs[0].pageContent.length);

// Text Splitter
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// Split the document into chunks
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

// Split the documents
const appSplitDocs = await splitter.splitDocuments(docs);

// console.log(appSplitDocs);
console.log(`Split blog post into ${appSplitDocs.length} sub-documents.`);

// Store the documents in the vector store
await vectorStore.addDocuments(appSplitDocs);

// ------------------------------------------------------
// 4️⃣ Retrieval and Generation (Multi-step approach where AI → retrieve tool → retrieve again → think → retrieve → answer)
// ------------------------------------------------------
import * as z from "zod";
import { tool } from "@langchain/core/tools";

const retrieveSchema = z.object({ query: z.string() });

const retrieve = tool(
  async ({ query }) => {
    console.log("Tool is running...");
    const retrievedDocs = await vectorStore.similaritySearch(query, 2);
    const serialized = retrievedDocs
      .map(
        (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`
      )
      .join("\n");
    return [serialized, retrievedDocs];
  },
  {
    name: "retrieve",
    description: "Retrieve information related to a query.",
    schema: retrieveSchema,
    responseFormat: "content_and_artifact",
  }
);

// ------------------------------------------------------
// 5️⃣ Agent
// ------------------------------------------------------
// import { createAgent } from "langchain";

const tools = [retrieve];

const systemPrompt =
  "You have access to a tool that retrieves context from a blog post. " +
  "Use the tool to help answer user queries.";

const agent = createAgent({ model, tools, systemPrompt });

let inputMessage = `What is the standard method for Task Decomposition?
Once you get the answer, look up common extensions of that method.`;

let agentInputs = { messages: [{ role: "user", content: inputMessage }] };

const response = await agent.invoke(agentInputs);
// console.log("Final Response:", response);

// const stream = await agent.stream(agentInputs, {
//   streamMode: "values",
// });
// for await (const step of stream) {
//   const lastMessage = step.messages[step.messages.length - 1];
//   console.log(`[${lastMessage.role}]: ${lastMessage.content}`);
//   console.log("-----\n");
// }

// ------------------------------------------------------
// 6️⃣ Retrieval and Generation (Two-step approach where AI → retrieve → answer)
// ------------------------------------------------------
import { createAgent, dynamicSystemPromptMiddleware } from "langchain";
import { SystemMessage } from "@langchain/core/messages";

// const agent1 = createAgent({
//   model,
//   tools: [],
//   middleware: [
//     dynamicSystemPromptMiddleware(async (state) => {
//       const lastQuery = state.messages[state.messages.length - 1].content;

//       const retrievedDocs = await vectorStore.similaritySearch(lastQuery as string, 2);

//       const docsContent = retrievedDocs
//         .map((doc) => doc.pageContent)
//         .join("\n\n");

//       // Build system message
//       const systemMessage = new SystemMessage(
//         `You are a helpful assistant. Use the following context in your response:\n\n${docsContent}`
//       );

//       // Return system + existing messages
//       return [systemMessage, ...state.messages];
//     }),
//   ],
// });

// let inputMessage1 = `What is Task Decomposition?`;

// let chainInputs = { messages: [{ role: "user", content: inputMessage1 }] };

// const stream = await agent1.stream(chainInputs, {
//   streamMode: "values",
// });
// for await (const step of stream) {
//   const lastMessage = step.messages[step.messages.length - 1];
//   console.log(`[${lastMessage.role}]: ${lastMessage.content}`);
//   console.log("-----\n");
// }
