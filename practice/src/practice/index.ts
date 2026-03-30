import { HumanMessage } from "@langchain/core/messages";
import { app } from "./agent.js";

(async () => {
  const result = await app.invoke({
    messages: [new HumanMessage("12 * 8 + 10 = ?")],
  });
  console.log("Test OK:", result.messages.at(-1)?.content);
})();
