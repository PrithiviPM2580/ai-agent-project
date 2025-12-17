import StateAnnotation from "../config/state.config.js";
import llm from "../config/llm.config.js";
import { missingInfoPrompt } from "../utils/index.util.js";
import { missingInfoSchema } from "../validator/index.validator.js";


async function missingInfoNode(state:typeof StateAnnotation.State){

    const promptQuery= missingInfoPrompt(state);

    const structuredLLM= llm.withStructuredOutput(missingInfoSchema);

    const response= await structuredLLM.invoke(promptQuery);

    return {
        missingInfo:response.missingInfo
    }
}

export default missingInfoNode;