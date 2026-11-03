import { describe, expect, it } from "vitest";
import type { StudyView } from "./api";
import { reportStatus } from "./study";

const view: StudyView = {
  study_id: "s1",
  status: null,
  stage: null,
  error: null,
  stages_done: [],
  artifacts: { design: "/studies/s1/artifacts/design" },
};

describe("reportStatus", () => {
  it("names the stage a failed report job stopped at, not the last one it finished", () => {
    expect(
      reportStatus({
        ...view,
        status: "failed",
        stage: "report",
        error: "the agent kept calling tools",
        stages_done: ["stats"],
      }),
    ).toBe("failed at report: the agent kept calling tools");
  });

  it("says when no report has been asked for, and when one is ready", () => {
    expect(reportStatus(view)).toBe("not asked for yet");
    expect(
      reportStatus({
        ...view,
        status: "done",
        artifacts: { ...view.artifacts, report: "/studies/s1/artifacts/report" },
      }),
    ).toBe("ready");
  });
});
