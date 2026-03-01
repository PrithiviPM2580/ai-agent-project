import {
  StateSchema,
  StateGraph,
  ReducedValue,
  GraphNode,
  Send,
  ConditionalEdgeRouter,
} from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import z from "zod/v4";
import { extractContent } from "src/utils/index.js";

const sectionsSchema = z.object({
  sections: z.array(
    z.object({
      name: z.string(),
      description: z.string(),
    }),
  ),
});

type SectionSchema = {
  name: string;
  description: string;
};

type SectionsSchema = z.infer<typeof sectionsSchema>;

//LLM model
const llm = new ChatGoogleGenerativeAI({
  temperature: 0.6,
  apiKey: process.env.GEMINI_API_KEY!,
  model: "gemini-2.5-flash",
});

const planner = llm.withStructuredOutput(sectionsSchema);

//StateSchema

const State = new StateSchema({
  topic: z.string(),
  sections: z.array(z.custom<SectionsSchema>()),
  completedSections: new ReducedValue(
    z.array(z.string()).default(() => []),
    { reducer: (a, b) => a.concat(b) },
  ),
  finalReport: z.string(),
});

const WorkerState = new StateSchema({
  section: z.custom<SectionSchema>(),
  completedSections: new ReducedValue(
    z.array(z.string()).default(() => []),
    { reducer: (a, b) => a.concat(b) },
  ),
});

//Nodes
const orchestratorNode: GraphNode<typeof State> = async (state, config) => {
  const reportSectionsRaw = await planner.invoke([
    { role: "system", content: "Generate a plan for the report" },
    { role: "human", content: `Here is the report topic: ${state.topic}` },
  ]);

  const reportSections: SectionSchema[] = JSON.parse(
    extractContent(reportSectionsRaw),
  );

  return {
    sections: reportSections,
  };
};

const llmcall: GraphNode<typeof WorkerState> = async (state, config) => {
  const section = await llm.invoke([
    {
      role: "system",
      content:
        "Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting.",
    },
    {
      role: "human",
      content: `Here is the section name: ${state.section.name} and description: ${state.section.description}`,
    },
  ]);

  return {
    completedSections: [extractContent(section)],
  };
};

const synthesizer: GraphNode<typeof State> = async (state) => {
  const completedSections = state.completedSections;

  const completedReportSections = completedSections.join("\n\n---\n\n");

  return {
    finalReport: completedReportSections,
  };
};

const assignWorkers: ConditionalEdgeRouter<typeof State, "llmCall"> = async (
  state,
) => {
  return state.sections.map((section) => {
    return new Send("llmCall", { section });
  });
};

//Graph
const orchestratorGraph = new StateGraph(State)
  .addNode("orchestrator", orchestratorNode)
  .addNode("llmCall", llmcall)
  .addNode("synthesizer", synthesizer)
  .addEdge("__start__", "orchestrator")
  .addConditionalEdges("orchestrator", assignWorkers, ["llmCall"])
  .addEdge("llmCall", "synthesizer")
  .addEdge("synthesizer", "__end__")
  .compile();

const state = await orchestratorGraph.invoke({
  topic: "Create a report on LLM scaling laws",
});

console.log("Final Report: ", state.finalReport);
