import { analyzeLogFile } from "./logAnalysis.js";

self.onmessage = (event) => {
  try {
    self.postMessage({
      type: "complete",
      report: analyzeLogFile(event.data.text, event.data.fileName),
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : "Unable to analyze this log file.",
    });
  }
};
