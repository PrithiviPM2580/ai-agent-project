import readline from "node:readline";
import { StateGraph } from "@langchain/langgraph";
import { plannerNode } from "./planner.js";

//CLI Input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
