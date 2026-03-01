export function extractContent(response: any): string {
  if (typeof response.content === "string") {
    return response.content;
  }
  if (Array.isArray(response.content)) {
    return response.content
      .map((c: any) => ("text" in c ? (c as any).text : String(c)))
      .join("");
  }
  return String(response.content);
}
