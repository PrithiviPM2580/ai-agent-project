// ============================================
//  🔹 RAG Chatbot
// ============================================

// ------------------------------------------------------
// 1️⃣ Components (Google Gemini)
// ------------------------------------------------------
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import "dotenv/config";

const model = new ChatGoogleGenerativeAI({
  model: "gemini-flash-2.5",
  apiKey: process.env.GOOGLE_API_KEY,
});
