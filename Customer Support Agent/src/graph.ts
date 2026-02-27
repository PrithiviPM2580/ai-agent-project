//Graph
import { START, END, StateGraph } from "@langchain/langgraph";
import { classifyIntent, draftResponse, humanReview } from "./nodes.js";
import { EmailAgentState } from "./state.js";

// Graph Construction
export function buildGraph() {
  const workflow = new StateGraph(EmailAgentState)
    .addNode("classifyIntent", classifyIntent, {
      ends: ["draftResponse", "humanReview"],
    })
    .addNode("draftResponse", draftResponse, {
      ends: [END],
    })
    .addNode("humanReview", humanReview, {
      ends: [END],
    })
    .addEdge(START, "classifyIntent")
    .addEdge("draftResponse", END)
    .addEdge("humanReview", END);

  return workflow.compile();
}
