import StateAnnotation from "../config/state.config.js";
import { readerPrompt } from "../utils/index.util.js";
import llm from "../config/llm.config.js";
import { readerSchema } from "../validator/index.validator.js";

async function readerNode(state: typeof StateAnnotation.State) {

    const promptQuery = readerPrompt(state);

    const structuredLLM = llm.withStructuredOutput(readerSchema);

    const response = await structuredLLM.invoke(promptQuery);

    return {
        summary: response.summary,
        sentiment: response.sentiment
    }

}

export default readerNode;