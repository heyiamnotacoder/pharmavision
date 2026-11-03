import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Testing Library unmounts after each test only when the runner has globals; Vitest's are off.
afterEach(cleanup);
