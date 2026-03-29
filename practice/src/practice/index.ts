import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { MODEL } from "src/constant/index.js";
import {
  StateGraph,
  StateSchema,
  MessagesValue,
  ReducedValue,
  GraphNode,
  ConditionalEdgeRouter,
  START,
  END,
} from "@langchain/langgraph";
import {
  AIMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";

// Model
const model = new ChatGoogleGenerativeAI({
  model: MODEL.GEMINI_2_5_FLASH,
  temperature: 0.5,
  apiKey: process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY,
});

// Tools
const add = tool(({ a, b }) => a + b, {
  name: "add",
  description: "Add two numbers",
  schema: z.object({
    a: z.number().describe("The first number to add"),
    b: z.number().describe("The second number to add"),
  }),
});

const subtract = tool(({ a, b }) => a - b, {
  name: "subtract",
  description: "Subtract two numbers",
  schema: z.object({
    a: z.number().describe("The first number to subtract"),
    b: z.number().describe("The second number to subtract"),
  }),
});

const multiply = tool(({ a, b }) => a * b, {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number().describe("The first number to multiply"),
    b: z.number().describe("The second number to multiply"),
  }),
});

const divide = tool(({ a, b }) => a / b, {
  name: "divide",
  description: "Divide two numbers",
  schema: z.object({
    a: z.number().describe("The first number to divide"),
    b: z.number().describe("The second number to divide"),
  }),
});

const toolsByName = {
  [add.name]: add,
  [subtract.name]: subtract,
  [multiply.name]: multiply,
  [divide.name]: divide,
};

const tools = Object.values(toolsByName);
const modelWithTools = model.bindTools(tools);

// State Schema
const MessagesState = new StateSchema({
  messages: MessagesValue,
  llmCalls: new ReducedValue(
    z.number().default(0).describe("The number of calls made to the LLM"),
    {
      reducer: (x, y) => x + y,
    },
  ),
});

// Graph Nodes
const llmCall: GraphNode<typeof MessagesState> = async (state) => {
  const response = await modelWithTools.invoke([
    new SystemMessage(
      "You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
    ),
    ...state.messages,
  ]);

  return {
    messages: [response],
    llmCalls: 1,
  };
};

const toolNode: GraphNode<typeof MessagesState> = async (state) => {
  const lastMessage = state.messages.at(-1);
  console.log("Last message:", lastMessage);
  if (lastMessage == null || !AIMessage.isInstance(lastMessage)) {
    return {
      messages: [],
    };
  }

  const result: ToolMessage[] = [];
  for (const toolCall of lastMessage.tool_calls ?? []) {
    const tool = toolsByName[toolCall.name as keyof typeof toolsByName];
    const observation = await tool.invoke(toolCall);
    result.push(observation);
  }

  return {
    messages: result,
  };
};

//Graph Workflow
const graph = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge(START, "llmCall")
  .addEdge("llmCall", "toolNode")
  .addEdge("toolNode", END)
  .compile();

// Execute the graph with an initial message
(async () => {
  const result = await graph.invoke({
    messages: [{ role: "user", content: "Who is the father of computer?" }],
  });
  console.log("Final Result:", result);
})();
