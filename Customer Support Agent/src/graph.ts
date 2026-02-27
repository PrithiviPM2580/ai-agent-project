//Graph
import { START, END, StateGraph } from "@langchain/langgraph";
import { classifyIntent, draftResponse } from "./nodes.js";
import { EmailAgentState } from "./state.js";

// Graph Construction
export function buildGraph() {
  const workflow = new StateGraph(EmailAgentState)
    .addNode("classifyIntent", classifyIntent)
    .addNode("draftResponse", draftResponse)
    .addEdge(START, "classifyIntent")
    .addEdge("classifyIntent", "draftResponse")
    .addEdge("draftResponse", END);

  return workflow.compile();
}
