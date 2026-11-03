/** `n` and the noun, plural unless n is 1: "1 warning", "2 warnings". */
export function plural(n: number, noun: string): string {
  return `${String(n)} ${noun}${n === 1 ? "" : "s"}`;
}
