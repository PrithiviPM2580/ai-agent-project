import { StateSchema, MessagesValue, GraphNode } from "@langchain/langgraph";

const State = new StateSchema({
  input: MessagesValue,
});

const plannerPrompt = `You are an expert learning strategist.

Break the user's goal into:
1. 4 major milestones
2. Weekly breakdown
3. Daily action steps

Keep it practical and project-based.
`;

export const plannerNode: GraphNode<typeof State> = (state) => {
  return {
    input: [{ role: "system", content: plannerPrompt }],
  };
};
