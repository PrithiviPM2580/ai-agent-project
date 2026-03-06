import "dotenv/config";
import { StateGraph, GraphNode, StateSchema } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";

//LLM model
const llm = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY!,
  model: "gemini-2.5-flash",
});

const State = new StateSchema({
  topic: z.string(),
  joke: z.string(),
  feedback: z.string(),
  funnyOrNot: z.string(),
});

const feedbackSchema = z.object({
  grade: z
    .enum(["funny", "not funny"])
    .describe("Grade the joke as either 'funny' or 'not funny'"),
  feedback: z
    .string()
    .describe(
      "Provide feedback on why you graded the joke as funny or not funny",
    ),
});

const evaluator = llm.withStructuredOutput(feedbackSchema);

//Graph nodes
const generateJoke: GraphNode<typeof State> = async (state) => {
  let msg;

  if (state.feedback) {
    msg = await llm.invoke(
      `Based on the feedback: ${state.feedback}, generate a joke about ${state.topic}.`,
    );
  } else {
    msg = await llm.invoke(`Generate a joke about ${state.topic}.`);
  }

  return {
    joke: msg.content.toString(),
  };
};

const evulateJoke: GraphNode<typeof State> = async (state) => {
  const grade = await evaluator.invoke(
    `Grade the joke ${state.joke} as either funny or not funny and provide feedback.`,
  );

  return {
    feedback: grade.feedback,
    funnyOrNot: grade.grade,
  };
};

type WorkflowState = Parameters<GraphNode<typeof State>>[0];

const checkIfFunny = async (
  state: WorkflowState,
): Promise<"Accepted" | "Rejected + Feedback"> => {
  if (state.funnyOrNot === "funny") {
    return "Accepted";
  } else {
    return "Rejected + Feedback";
  }
};

//Build workflow

const workflow = new StateGraph(State)
  .addNode("generateJoke", generateJoke)
  .addNode("evulateJoke", evulateJoke)
  .addEdge("__start__", "generateJoke")
  .addEdge("generateJoke", "evulateJoke")
  .addConditionalEdges("evulateJoke", checkIfFunny, {
    Accepted: "__end__",
    "Rejected + Feedback": "generateJoke",
  })
  .compile();

//Run workflow
const result = await workflow.invoke({ topic: "programming" });
console.log("Final Result:", result);
console.log("Joke: ", result.joke);
