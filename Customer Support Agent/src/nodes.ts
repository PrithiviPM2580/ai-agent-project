//Nodes
import { HumanMessage } from "@langchain/core/messages";
import { EmailAgentState, EmailClassificationSchema } from "./state.js";
import { Command, GraphNode } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

//Model
const llm = new ChatGoogleGenerativeAI({
  temperature: 0.4,
  apiKey: process.env.GOOGLE_API_KEY!,
  model: "gemini-2.5-pro",
});

// Nodes

export const classifyIntent: GraphNode<typeof EmailAgentState> = async (
  state,
  config,
) => {
  const structuredLlm = llm.withStructuredOutput(EmailClassificationSchema);

  const prompt = `Classify this customer email:

    Email: ${state.emailContent}
    From: ${state.senderEmail}

    Provide intent, urgency, topic, and summary.
    `;
  const classification = await structuredLlm.invoke(prompt);

  return new Command({
    update: { classification },
    goto: "draftResponse",
  });
};

export const draftResponse: GraphNode<typeof EmailAgentState> = async (
  state,
  config,
) => {
  const { emailContent, classification } = state;

  const prompt = `
    Draft a professional response to:
    ${emailContent}

    Intent: ${classification?.intent}
    Urgency: ${classification?.urgency}

   Be helpful and professional.
  `;

  const response = await llm.invoke([new HumanMessage(prompt)]);

  return {
    responseText: response.content.toString(),
  };
};
