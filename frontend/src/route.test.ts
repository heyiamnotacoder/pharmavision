import { describe, expect, it } from "vitest";
import { runFromHash } from "./route";

describe("runFromHash", () => {
  it("reads the run id from #/runs/<id>", () => {
    expect(runFromHash("#/runs/20261103T053425Z-dc5242")).toBe("20261103T053425Z-dc5242");
  });

  it("decodes an escaped id", () => {
    expect(runFromHash("#/runs/OFT%2016")).toBe("OFT 16");
  });

  it("is null for the run list and for anything else", () => {
    expect(runFromHash("")).toBeNull();
    expect(runFromHash("#/")).toBeNull();
    expect(runFromHash("#/runs/")).toBeNull();
    expect(runFromHash("#/studies/s1")).toBeNull();
  });
});
