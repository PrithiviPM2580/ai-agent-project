import StateAnnotation from "../config/state.config.js";
import llm from "../config/llm.config.js";
import { teamMap } from "../utils/index.util.js";
import { index } from "@langchain/core/indexing";

type TeamKey = "Payment" | "Account" | "Technical" | "Delivery" | "Other"

async function routerNode(state: typeof StateAnnotation.State) {

    const category = state.category as TeamKey;
    return {
        assignedTeam: teamMap[category] || "General Support Team"
    }

};

export default routerNode;