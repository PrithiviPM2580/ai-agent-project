import { z } from "zod";

export const readerSchema = z.object({
  summary: z.string().describe("Concise summary of the customer support message"),
  sentiment: z.enum(["positive", "neutral", "negative"]).describe("Sentiment of the message"),
});


export const classifierSchema = z.object({
  category: z.enum(["Payment", "Account", "Technical", "Delivery", "Other"]).describe("Category of the customer support message").describe("Category of the customer support message")
});

export const prioritySchema = z.object({
  priority: z.enum(["low", "medium", "high"]).describe("Priority level of the customer support message")
});

export const missingInfoSchema = z.object({
  missingInfo: z.string().describe(
    'Return exactly "none" if nothing is missing, otherwise a short description'
  )
});

export const spamCheckSchema = z.object({
  isSpam: z.boolean().describe("Indicates if the message is spam or not"),
  spamType: z.enum(["spam", "scam", "phishing", "abuse", "none"]).describe("Type of spam detected")
})

export type ReaderOutput = z.infer<typeof readerSchema>;
export type ClassifierOutput = z.infer<typeof classifierSchema>;
export type PriorityOutput = z.infer<typeof prioritySchema>;
export type MissingInfoOutput = z.infer<typeof missingInfoSchema>;