//State, generate joke, checkpunlines, improve joke polish joke,

import {
  StateGraph,
  StateSchema,
  START,
  END,
  GraphNode,
  ConditionalEdgeRouter,
} from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import z from "zod/v4";

//Define LLM
const llm = new ChatGoogleGenerativeAI({
  temperature: 0.4,
  apiKey: process.env.GEMINI_API_KEY!,
  model: "gemini-2.5-flash",
});

// Define the state schema
const State = new StateSchema({
  topic: z.string().describe("The topic of the joke"),
  joke: z.string().describe("The joke to be improved"),
  improvedJoke: z.string().describe("The improved joke"),
  finalJoke: z.string().describe("The final polished joke"),
});

//Define Nodes
const generateJokeNode: GraphNode<typeof State> = async (state) => {
  const msg = await llm.invoke(`Write a short joke about ${state.topic}`);

  return {
    joke: msg.content,
  };
};

const checkPunchlineNode: ConditionalEdgeRouter<
  typeof State,
  "improveJokeNode"
> = async (state) => {
  if (state.joke.includes("?") || state.joke.includes("!")) {
    return "Pass";
  }
  return "Fail";
};

const improveJokeNode: GraphNode<typeof State> = async (state) => {
  const msg = await llm.invoke(
    `Make this joke funnier by adding wordplay: ${state.joke}`,
  );

  return {
    improvedJoke: msg.content,
  };
};

const polishJokeNode: GraphNode<typeof State> = async (state) => {
  const msg = await llm.invoke(
    `Polish this joke to make it more concise and impactful: ${state.improvedJoke}`,
  );

  return {
    finalJoke: msg.content,
  };
};

// Define the state graph
const workflow = new StateGraph(State)
  .addNode("generateJokeNode", generateJokeNode)
  .addNode("improveJokeNode", improveJokeNode)
  .addNode("polishJokeNode", polishJokeNode)
  .addEdge(START, "generateJokeNode")
  .addConditionalEdges("generateJokeNode", checkPunchlineNode, {
    Pass: "improveJokeNode",
    Fail: "__end__",
  })
  .addEdge("improveJokeNode", "polishJokeNode")
  .addEdge("polishJokeNode", END)
  .compile();

// Execute the graph
const initialState = {
  topic: "programming",
};

const result = await workflow.invoke(initialState);
console.log("Final Polished Joke:", result.finalJoke);
