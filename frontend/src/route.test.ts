import { describe, expect, it } from "vitest";
import { routeFromHash } from "./route";

describe("routeFromHash", () => {
  it("reads the run id from #/runs/<id> and the study id from #/studies/<id>", () => {
    expect(routeFromHash("#/runs/20261103T053425Z-dc5242")).toEqual({
      screen: "run",
      id: "20261103T053425Z-dc5242",
    });
    expect(routeFromHash("#/studies/s1")).toEqual({ screen: "study", id: "s1" });
  });

  it("decodes an escaped id", () => {
    expect(routeFromHash("#/runs/OFT%2016")).toEqual({ screen: "run", id: "OFT 16" });
  });

  it("is the list for the root and for anything else", () => {
    for (const hash of ["", "#/", "#/runs/", "#/studies/", "#/other/x"]) {
      expect(routeFromHash(hash)).toEqual({ screen: "list" });
    }
  });
});
