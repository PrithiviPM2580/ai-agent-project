import type StateAnnotation from "../config/state.config.js";

export function readerPrompt(state: typeof StateAnnotation.State): string {
    return `
     You are an AI assistant specialized in analyzing customer support messages.
     Given the following customer support message, please provide a concise
     summary, categorize the issue, determine its priority level (low, medium, high),
     and assess the sentiment (positive, neutral, negative). If any critical information
     is missing to perform these tasks, please identify what is needed.

     Message:
     ${state.originalMessage}
    `;
};

export function classifierPrompt(state: typeof StateAnnotation.State): string {
    return `
    Classify the customer issue.

    Summary:
    "${state.summary}"
    
    Categories:
    - Payment
    - Account
    - Technical
    - Delivery
    - Other
    `;
};

export function priorityPrompt(state: typeof StateAnnotation.State): string {
    return `
    Decide priority based on issue & sentiment.

    Category: ${state.category}
    Sentiment: ${state.sentiment}
    
    Rules:
    - Angry + Payment → High
    - Account locked → High
    - Everything else → Medium or Low
    `;
};

export function missingInfoPrompt(
  state: typeof StateAnnotation.State
): string {
  return `
You are a customer support triage assistant.

Based on the summary below, decide if a human support agent
can proceed WITHOUT asking the customer any follow-up questions.

Summary:
"${state.summary}"

Rules:
- If NO follow-up question is required → return "none"
- If a follow-up question IS required → return a short description of what is missing
- Return ONLY the value, no explanations
`;
};


export function spamCheckPrompt(state: typeof StateAnnotation.State): string {
  return `
  You are a spam detection system.

  Message:
  "${state.originalMessage}"
  
  Decide if this message is:
  - Spam (ads, promotions)
  - Scam (fraud, phishing)
  - Abuse (threats, harassment)
  - Legitimate
  `;
}





export const teamMap = {
    Payment: "Billing Team",
    Account: "Account Support",
    Technical: "Tech Support",
    Delivery: "Logistics",
    Other: "General Support"
} as const;