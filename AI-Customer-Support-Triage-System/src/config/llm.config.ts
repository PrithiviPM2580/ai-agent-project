import {ChatGoogleGenerativeAI} from "@langchain/google-genai";


const llm= new ChatGoogleGenerativeAI({
    model:"gemini-2.5-pro",
    apiKey: process.env.GEMINI_API_KEY!,
});

export default llm;