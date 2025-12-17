import StateAnnotation from "../config/state.config.js";
import llm from "../config/llm.config.js";
import { classifierSchema } from "../validator/index.validator.js";
import { classifierPrompt } from "../utils/index.util.js";

async function classifierNode(state: typeof StateAnnotation.State) {

    const promptQuery = classifierPrompt(state);

    const structuredLLM = llm.withStructuredOutput(classifierSchema);

    const response = await structuredLLM.invoke(promptQuery);

    return {
        category: response.category
    }
}

export default classifierNode;