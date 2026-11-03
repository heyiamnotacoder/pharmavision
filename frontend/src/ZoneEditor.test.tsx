import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Arena, Point } from "./api";
import { ZoneEditor } from "./ZoneEditor";

const FLOOR: Point[] = [
  [30, 30],
  [330, 30],
  [330, 330],
  [30, 330],
];
const detected: Arena = {
  floor_px: FLOOR,
  centre_px: [],
  px_per_cm: 300 / 45,
  confirmed: false,
};
const empty: Arena = { floor_px: [], centre_px: [], px_per_cm: 0, confirmed: false };

/** jsdom has no PointerEvent: without one, Testing Library's pointer events lose `clientX`. */
class TestPointerEvent extends MouseEvent {
  pointerId: number;
  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 1;
  }
}

let posted: { url: string; body: unknown }[];
let reply: Response;

beforeEach(() => {
  posted = [];
  reply = new Response(JSON.stringify({ status: "queued" }), { status: 202 });
  vi.stubGlobal("PointerEvent", TestPointerEvent);
  Element.prototype.setPointerCapture = () => undefined; // jsdom has no pointer capture
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    posted.push({ url, body: JSON.parse(init?.body as string) });
    return Promise.resolve(reply);
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function editor(arena: Arena, onConfirmed = vi.fn()) {
  const view = render(
    <ZoneEditor
      runId="r1"
      backgroundUrl="/api/runs/r1/background"
      width={360}
      height={360}
      arena={arena}
      centreFraction={0.5}
      onConfirmed={onConfirmed}
    />,
  );
  const svg = view.container.querySelector("svg");
  if (!svg) throw new Error("no editor drawn");
  // Drawn at its pixel size, so a pointer's client position is its video pixel.
  svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 360, height: 360 }) as DOMRect;
  const centre = () => view.container.querySelector("polygon.centre")?.getAttribute("points");
  return { svg, centre, onConfirmed };
}

function drag(svg: SVGSVGElement, handle: Element, to: Point) {
  fireEvent.pointerDown(handle, { clientX: 0, clientY: 0, pointerId: 1 });
  fireEvent.pointerMove(svg, { clientX: to[0], clientY: to[1], pointerId: 1 });
  fireEvent.pointerUp(svg, { clientX: to[0], clientY: to[1], pointerId: 1 });
}

describe("ZoneEditor", () => {
  it("sends the floor with a dragged corner when the zones are confirmed", async () => {
    const { svg, onConfirmed } = editor(detected);

    drag(svg, screen.getByRole("button", { name: /^floor corner 3/ }), [300, 300]);
    fireEvent.click(screen.getByRole("button", { name: "Confirm zones" }));

    await waitFor(() => {
      expect(onConfirmed).toHaveBeenCalled();
    });
    expect(posted).toEqual([
      {
        url: "/api/runs/r1/arena/confirm",
        body: { floor_px: [FLOOR[0], FLOOR[1], [300, 300], FLOOR[3]] },
      },
    ]);
  });

  it("moves the centre zone with the floor, and gives the centre no corners of its own", () => {
    const { svg, centre } = editor(detected);
    const before = centre();

    drag(svg, screen.getByRole("button", { name: /^floor corner 3/ }), [300, 300]);

    expect(centre()).not.toEqual(before);
    expect(screen.getAllByRole("button", { name: /corner/ })).toHaveLength(4);
  });

  it("moves a focused corner with the arrow keys", () => {
    editor(detected);
    const corner = screen.getByRole("button", { name: /^floor corner 1/ });

    fireEvent.keyDown(corner, { key: "ArrowRight" });
    fireEvent.keyDown(corner, { key: "ArrowDown", shiftKey: true });

    expect(screen.getByRole("button", { name: /^floor corner 1/ })).toHaveAccessibleName(
      "floor corner 1 at 31, 40 px",
    );
  });

  it("has the floor drawn corner by corner where none was found", async () => {
    const { svg } = editor(empty);
    const confirm = screen.getByRole("button", { name: "Confirm zones" });
    expect(screen.getByText(/found no floor/)).toBeInTheDocument();
    expect(confirm).toBeDisabled();

    for (const [x, y] of FLOOR) fireEvent.pointerDown(svg, { clientX: x, clientY: y });
    fireEvent.pointerDown(svg, { clientX: 100, clientY: 100 }); // a fifth click adds nothing
    fireEvent.click(confirm);

    await waitFor(() => {
      expect(posted).toHaveLength(1);
    });
    expect(posted[0]?.body).toEqual({ floor_px: FLOOR });
  });

  it("shows why a floor was refused and stays open", async () => {
    reply = new Response(JSON.stringify({ detail: "the floor's edges cross" }), { status: 422 });
    const { onConfirmed } = editor(detected);

    fireEvent.click(screen.getByRole("button", { name: "Confirm zones" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("the floor's edges cross");
    expect(onConfirmed).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Confirm zones" })).toBeEnabled();
  });
});
