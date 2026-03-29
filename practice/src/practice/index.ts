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

// Model
const model = new ChatGoogleGenerativeAI({
  model: MODEL.GEMINI_2_5_FLASH,
  temperature: 0.5,
  apiKey: process.env.GEMINI_API_KEY,
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

const MessagesState = new StateSchema({
  messages: MessagesValue,
  llmCalls: new ReducedValue(
    z.number().default(0).describe("The number of calls made to the LLM"),
    {
      reducer: (x, y) => x + y,
    },
  ),
});
