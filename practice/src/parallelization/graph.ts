import "dotenv/config";
import { StateGraph, StateSchema, GraphNode } from "@langchain/langgraph";
import z from "zod/v4";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

//LLM Model
const llm = new ChatGoogleGenerativeAI({
  temperature: 0.5,
  apiKey: process.env.GEMINI_API_KEY!,
  model: "gemini-2.5-flash",
});

//State Schema
const State = new StateSchema({
  topic: z.string(),
  joke: z.string(),
  story: z.string(),
  poem: z.string(),
  combinedOutput: z.string(),
});

//Graph Nodes
const chatLllm1: GraphNode<typeof State> = async (state) => {
  const result = await llm.invoke(`Write a joke about ${state.topic}`);
  return {
    joke: result.content.toString(),
  };
};

const chatLlm2: GraphNode<typeof State> = async (state) => {
  const result = await llm.invoke(`Write a story about ${state.topic}`);
  return {
    story: result.content.toString(),
  };
};

const chatLlm3: GraphNode<typeof State> = async (state) => {
  const result = await llm.invoke(`Write a poem about ${state.topic}`);
  return {
    poem: result.content.toString(),
  };
};

const aggregator: GraphNode<typeof State> = async (state) => {
  console.log("Aggregating results...");
  console.log("Current state:", state);
  const combined =
    `Here's a story, joke, and poem about ${state.topic}!\n\n` +
    `STORY:\n${state.story}\n\n` +
    `JOKE:\n${state.joke}\n\n` +
    `POEM:\n${state.poem}`;

  return {
    combinedOutput: combined,
  };
};

//Build Workflow
const parallelGraph = new StateGraph(State)
  .addNode("chatLllm1", chatLllm1)
  .addNode("chatLlm2", chatLlm2)
  .addNode("chatLlm3", chatLlm3)
  .addNode("aggregator", aggregator)
  .addEdge("__start__", "chatLllm1")
  .addEdge("__start__", "chatLlm2")
  .addEdge("__start__", "chatLlm3")
  .addEdge("chatLllm1", "aggregator")
  .addEdge("chatLlm2", "aggregator")
  .addEdge("chatLlm3", "aggregator")
  .addEdge("aggregator", "__end__")
  .compile();

const result = await parallelGraph.invoke({ topic: "cats" });
console.log("Final Result:", result.combinedOutput);
