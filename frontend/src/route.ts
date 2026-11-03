export type Route = { screen: "list" } | { screen: "run" | "study"; id: string };

/** `#/runs/<id>` opens a run, `#/studies/<id>` a study's report; anything else lists them. No
 * router needed for three screens. */
export function routeFromHash(hash: string): Route {
  const match = /^#\/(runs|studies)\/(.+)$/.exec(hash);
  if (!match?.[1] || !match[2]) return { screen: "list" };
  return { screen: match[1] === "runs" ? "run" : "study", id: decodeURIComponent(match[2]) };
}
