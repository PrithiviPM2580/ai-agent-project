import "dotenv/config";
import { task, entrypoint } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

//LLM Model
const llm = new ChatGoogleGenerativeAI({
  temperature: 0.4,
  apiKey: process.env.GEMINI_API_KEY!,
  model: "gemini-2.5-flash",
});

//Nodes

const callLlm1 = task("callLlm1", async (topic: string) => {
  const result = await llm.invoke(`Write a story about ${topic}?`);
  return result.content.toString();
});

const callLlm2 = task("callLlm2", async (topic: string) => {
  const result = await llm.invoke(`Write a poem about ${topic}?`);
  return result.content.toString();
});

const callLlm3 = task("callLlm3", async (topic: string) => {
  const result = await llm.invoke(`Write a joke about ${topic}?`);
  return result.content.toString();
});

const aggregator = (params: {
  topic: string;
  story: string;
  poem: string;
  joke: string;
}) => {
  const { topic, story, poem, joke } = params;
  return (
    `Here's a story, joke, and poem about ${topic}!\n\n` +
    `STORY:\n${story}\n\n` +
    `JOKE:\n${joke}\n\n` +
    `POEM:\n${poem}`
  );
};

//Workflow
const workflow = entrypoint("workflow", async (topic: string) => {
  const [story, poem, joke] = await Promise.all([
    callLlm1(topic),
    callLlm2(topic),
    callLlm3(topic),
  ]);
  return aggregator({ topic, story, poem, joke });
});

const response = await workflow.invoke("cats");
console.log("Final Response:\n", response);
