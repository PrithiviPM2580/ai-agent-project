import StateAnnotation from "../config/state.config.js";
import llm from "../config/llm.config.js";
import { priorityPrompt } from "../utils/index.util.js";
import { prioritySchema } from "../validator/index.validator.js";


async function priorityNode(state: typeof StateAnnotation.State) {

    const promptQuery = priorityPrompt(state);

    const structuredLLM = llm.withStructuredOutput(prioritySchema);

    const response = await structuredLLM.invoke(promptQuery);

    return {
        priority:response.priority
    }
};

export default priorityNode;