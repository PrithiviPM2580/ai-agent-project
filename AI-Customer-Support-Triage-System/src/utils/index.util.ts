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
Check if important information is missing.

Summary:
"${state.summary}"

Rules (VERY IMPORTANT):
- If no information is missing → return exactly: "none"
- If information is missing → return ONLY a short description
- Do NOT return explanations
- Do NOT return JSON code fences
`;
}




export const teamMap = {
    Payment: "Billing Team",
    Account: "Account Support",
    Technical: "Tech Support",
    Delivery: "Logistics",
    Other: "General Support"
} as const;