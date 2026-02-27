//Define State

import { StateSchema } from "@langchain/langgraph";
import * as z from "zod";

// Define the schema for email classification
export const EmailClassificationSchema = z.object({
  intent: z.enum(["question", "bug", "billing", "feature", "complex"]),
  urgency: z.enum(["low", "medium", "high", "critical"]),
  topic: z.string(),
  summary: z.string(),
});

// Define the state schema for the email agent
export const EmailAgentState = new StateSchema({
  emailContent: z.string(),
  senderEmail: z.string(),

  classification: EmailClassificationSchema.optional(),
  responseText: z.string().optional(),
});

// Define the type for email classification
export type EmailClassificationType = z.infer<typeof EmailClassificationSchema>;
