export const MODEL = {
  GEMINI_2_5_PRO: "gemini-2.5-pro",
  GEMINI_2_5_FLASH: "gemini-2.5-flash",
  GEMINI_2_5_FLASH_PREVIEW_09_2025: "gemini-2.5-flash-preview-09-2025",
  GEMINI_2_5_FLASH_LITE: "gemini-2.5-flash-lite",
  GEMINI_2_5_FLASH_LITE_PREVIEW_09_2025:
    "gemini-2.5-flash-lite-preview-09-2025",
} as const;

export type MODEL_TYPE = (typeof MODEL)[keyof typeof MODEL];
