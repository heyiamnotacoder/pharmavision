/** `#/runs/<id>` opens a run; anything else lists them. No router needed for two screens. */
export function runFromHash(hash: string): string | null {
  const match = /^#\/runs\/(.+)$/.exec(hash);
  return match?.[1] ? decodeURIComponent(match[1]) : null;
}
