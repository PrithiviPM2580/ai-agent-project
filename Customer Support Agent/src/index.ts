//Main
import { buildGraph } from "./graph.js";

async function main() {
  const app = buildGraph();

  const result = await app.invoke({
    emailContent: "I was charged twice for my subscription!",
    senderEmail: "customer@email.com",
  });

  console.log("\n--- FINAL RESPONSE ---\n");
  console.log(result.responseText);
}
