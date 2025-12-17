import StateAnnotation from "../config/state.config.js";
import llm from "../config/llm.config.js";
import { spamCheckPrompt } from "../utils/index.util.js";
import { spamCheckSchema } from "../validator/index.validator.js";


async function spamCheckNode(state:typeof StateAnnotation.State){

    const promptQuery= spamCheckPrompt(state);

    const structuredLLM= llm.withStructuredOutput(spamCheckSchema);

    const response= await structuredLLM.invoke(promptQuery);

    return {
        isSpam:response.isSpam,
        spamType:response.spamType
    }

};

export default spamCheckNode;