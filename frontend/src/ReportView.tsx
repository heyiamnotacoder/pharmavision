import type { Nodes, Root } from "mdast";
import { findAndReplace } from "mdast-util-find-and-replace";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Report } from "./api";
import { plural } from "./text";

/** As `arrive.placeholder` writes them. */
const TO_FILL = /\[TO FILL: [^\]\n]*\]/g;
/** `arrive.STRUCK_FLAG`, inside its bold. */
const STRUCK_FLAG = "[failed the number audit]";

/** Tags what the report stage marked, so it renders apart from the prose the audit passed:
 * each `[TO FILL: …]` placeholder, each struck-out (`~~…~~`) sentence and its flag. */
function remarkReportMarks() {
  return (tree: Root) => {
    findAndReplace(tree, [
      TO_FILL,
      (match: string) => ({
        type: "emphasis",
        children: [{ type: "text", value: match }],
        data: { hName: "mark", hProperties: { className: ["to-fill"] } },
      }),
    ]);
    const walk = (node: Nodes) => {
      if (node.type === "delete") node.data = { hProperties: { className: ["struck"] } };
      if (
        node.type === "strong" &&
        node.children.length === 1 &&
        node.children[0]?.type === "text" &&
        node.children[0].value === STRUCK_FLAG
      ) {
        node.data = { hProperties: { className: ["audit-flag"] } };
      }
      if ("children" in node) node.children.forEach(walk);
    };
    walk(tree);
  };
}

/** The last draft's number audit and the placeholders left, in one line. */
function auditSummary(report: Report): string {
  const numbers = report.drafts.at(-1)?.audit.numbers ?? [];
  const failed = numbers.filter((n) => n.source === null);
  const sentences = new Set(failed.map((n) => n.sentence)).size;
  const toFill = report.markdown.match(TO_FILL)?.length ?? 0;
  const audit = failed.length
    ? `Number audit failed: ${String(failed.length)} of ${plural(numbers.length, "number")} ` +
      `matched no computed value, so ${plural(sentences, "sentence")} ` +
      `${sentences === 1 ? "is" : "are"} struck out.`
    : `Number audit passed: all ${plural(numbers.length, "number")} match a computed value.`;
  return `${audit} ${plural(toFill, "item")} to fill.`;
}

export function ReportView({ report }: { report: Report }) {
  return (
    <article className="report">
      <p role="status" className="audit">
        {auditSummary(report)}
      </p>
      <Markdown remarkPlugins={[remarkGfm, remarkReportMarks]}>{report.markdown}</Markdown>
      <p className="muted">
        Written by {report.model_id} from the computed tables after{" "}
        {plural(report.drafts.length, "draft")}; research use only, not validated for regulatory GLP
        studies.
      </p>
    </article>
  );
}
