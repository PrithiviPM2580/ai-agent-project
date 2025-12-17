# 🧠 AI Customer Support Triage System

## 🔴 Problem Statement

Companies receive hundreds of customer messages every day via email, chat, and forms. Before solving issues, human agents waste time **reading, understanding, categorizing, prioritizing, and routing** these messages.

The goal of this system is **not to solve customer problems**, but to **organize and route them intelligently** so humans can act faster.

---

## ❌ Current Problems

* Messages are **unstructured** (long, emotional, unclear)
* Human agents must:

  * Read the message
  * Understand intent
  * Decide urgency
  * Route to the correct team
* This causes **delays → unhappy customers 😡**

---

## 🎯 Goal of the AI System

Build an AI Agent System that can:

* Understand customer messages
* Classify the issue type
* Decide priority/urgency
* Route the issue to the correct team
* Ask follow-up questions if information is missing

⚠️ **Important Constraint**

> The AI does **NOT** resolve the issue. It only triages and organizes it.

---

## 🧩 Why This Is a Good LangGraph Problem

This problem **cannot** be solved with a single prompt.

It requires:

* Decisions
* Branching logic
* Looping
* Shared memory/state
* Multiple AI roles working together

➡️ This makes it **perfect for LangChain + LangGraph**.

---

## 🏗️ High-Level System Design (No Code)

### 🧠 AI Roles (Mental Model)

Think of the system as **multiple specialized AI agents**, not one giant model.

1. **Reader Agent**

   * Reads the raw customer message
   * Extracts structured information

2. **Classifier Agent**

   * Determines the type of issue (payment, technical, account, etc.)

3. **Priority Agent**

   * Decides urgency (Low / Medium / High / Critical)

4. **Router Agent**

   * Assigns the message to the correct internal team

5. **Clarification Agent**

   * Detects missing information
   * Asks follow-up questions

---

## 🔄 Flow Design (Core Logic)

```
User Message
   ↓
Reader Agent
   ↓
Classifier Agent
   ↓
Priority Agent
   ↓
 ┌────────────────┐
 │ Missing Info ? │
 └───────┬────────┘
         │ Yes
         ↓
Clarification Agent → User
         │
         └── Wait for reply → back to Reader Agent

         No
         ↓
Router Agent
   ↓
Final Structured Output
```

This **branching + looping** behavior is exactly why LangGraph is ideal.

---

## 🗂️ Structured Output (Final Result)

The system produces a clean, machine-readable output for humans and tools:

```json
{
  "category": "Payment Issue",
  "priority": "High",
  "sentiment": "Angry",
  "assigned_team": "Billing Team",
  "summary": "Customer was charged twice for the same order.",
  "needs_human": true
}
```

---

## 🧠 State Design (Conceptual)

The system maintains a **shared state** that all agents can read and update.

### Core State Fields

* `original_message`
* `extracted_info`
* `category`
* `priority`
* `sentiment`
* `missing_fields`
* `assigned_team`
* `final_decision`

👉 Each LangGraph node **reads from state and writes back to it**.

---

## 🎛️ Real-World Value

This type of system is used in:

* SaaS companies
* Banks
* E-commerce platforms
* Telecom providers
* Government service portals

You are practicing **AI system architecture**, not just prompt engineering.

---

## 🧪 What You Will Learn

* How to decompose problems into AI agents
* When to branch vs loop in workflows
* How state flows in LangGraph
* How AI systems behave like backend pipelines
* How real startups design AI products

---

## ✅ Next Design Step

Choose **ONE** direction to continue:

1️⃣ Draw the LangGraph nodes & edges (visual explanation)
2️⃣ Design the state schema in deep detail
3️⃣ Write prompts for each agent (still no code)
4️⃣ Add real-world edge cases (angry users, spam, empty messages)

➡️ Reply wit
