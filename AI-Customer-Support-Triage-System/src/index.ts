import "dotenv/config";
import { START, END, StateGraph } from "@langchain/langgraph";
import StateAnnotation from "./config/state.config.js";
import readerNode from "./nodes/readerAgent.node.js";
import classifierNode from "./nodes/classifierAgent.node.js";
import priorityNode from "./nodes/priorityAgent.node.js";
import missingInfoNode from "./nodes/missingInfoAgent.node.js";
import routerNode from "./nodes/routerAgent.node.js";

const builder = new StateGraph(StateAnnotation)
    .addNode("readerNode", readerNode)
    .addNode("classifierNode", classifierNode)
    .addNode("priorityNode", priorityNode)
    .addNode("missingInfoNode", missingInfoNode)
    .addNode("routerNode", routerNode)
    .addEdge(START, "readerNode")
    .addEdge("readerNode", "classifierNode")
    .addEdge("classifierNode", "priorityNode")
    .addEdge("priorityNode", "missingInfoNode")
    .addConditionalEdges("missingInfoNode", (state) => {
        if (state.missingInfo !== "none") {
            console.log("Critical information is missing:", state.missingInfo);
            return END
        }

        console.log("All necessary information is present. Proceeding to routing.");
        return "routerNode"
    })
    .addEdge("routerNode", END);

const graph = builder.compile();

const query = "Can you help me reset my password? I forgot my current one and need to access my account.";
console.log("User Query:", query);

const result = await graph.invoke({
    originalMessage: query
});

console.log("Final Result:", result);