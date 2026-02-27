import "dotenv/config";
import { task, entrypoint } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

//Define LLM
const llm = new ChatGoogleGenerativeAI({
  temperature: 0.4,
  apiKey: process.env.GEMINI_API_KEY!,
  model: "gemini-2.5-flash",
});

const generateJoke = task("generateJoke", async (topic: string) => {
  const mgs = await llm.invoke(`Write a short joke about ${topic}`);

  return mgs.content as string;
});

const checkPunchline = (joke: string) => {
  if (joke.includes("?") || joke.includes("!")) {
    return "PASS";
  }
  return "FAIL";
};

const improveJoke = task("improveJoke", async (joke: string) => {
  const mgs = await llm.invoke(
    `Improve the following joke with the wordplay: ${joke}`,
  );
  return mgs.content as string;
});

const polishJoke = task("polishJoke", async (joke: string) => {
  const mgs = await llm.invoke(`Polish the following joke: ${joke}`);
  return mgs.content as string;
});

const workflow = entrypoint("jokeWorkflow", async (topic: string) => {
  const originalJoke = await generateJoke(topic);
  if (checkPunchline(originalJoke) === "PASS") {
    return originalJoke;
  }

  const improvedJoke = await improveJoke(originalJoke);
  const polishedJoke = await polishJoke(improvedJoke);
  return polishedJoke;
});

const joke = await workflow.invoke("programming");
console.log("Final Joke: ", joke);
