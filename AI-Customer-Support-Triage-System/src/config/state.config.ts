import { Annotation } from "@langchain/langgraph";


const StateAnnotation = Annotation.Root({
    originalMessage: Annotation(),
    isSpam: Annotation(),
    spamType: Annotation(),
    summary: Annotation(),
    category: Annotation(),
    priority: Annotation(),
    sentiment: Annotation(),
    missingInfo: Annotation(),
    assignedTeam: Annotation(),
});


export default StateAnnotation;