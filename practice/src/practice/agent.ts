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
  HumanMessage,
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
    new SystemMessage(`
You are a smart AI assistant that can either answer directly or use tools.

Decision rules:
- If the user asks an arithmetic or math-related question, use the appropriate tool to compute the answer.
- If the question involves calculations (especially multi-step or precise), ALWAYS use a tool.
- If the question is about real-world knowledge, explanations, opinions, or general conversation, respond directly without using tools.

Do NOT use tools for:
- General knowledge questions
- Explanations
- Conversations
- Conceptual questions

Do use tools for:
- Arithmetic (addition, subtraction, multiplication, division)
- Multi-step calculations
- Any query requiring precise numerical results

Behavior:
- After using a tool, return a clear and natural answer.
- Do not mention the tool unless necessary.
- If no tool is needed, just answer normally.
- Ask for clarification if the query is unclear.

You are efficient in choosing between thinking and tool usage.
`),
    ...state.messages,
  ]);

  return {
    messages: [response],
    llmCalls: 1,
  };
};

const toolNode: GraphNode<typeof MessagesState> = async (state) => {
  const lastMessage = state.messages.at(-1);
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

const shouldContinue: ConditionalEdgeRouter<
  typeof MessagesState,
  Record<string, any>,
  "toolNode"
> = (state) => {
  const lastMessage = state.messages.at(-1);

  if (!lastMessage || !AIMessage.isInstance(lastMessage)) {
    return END;
  }

  if (lastMessage.tool_calls?.length) {
    return "toolNode";
  }
  return END;
};

//Graph Workflow
const graph = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])
  .addEdge("toolNode", "llmCall");

export const app = graph.compile();

export function makeAgent() {
  return app;
}
