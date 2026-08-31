import { describe, expect, it } from "vitest";
import { deltaColor } from "./TransactionFlow.jsx";

describe("deltaColor", () => {
  it("returns red when the balance decreased", () => {
    expect(deltaColor(1000, 500)).toBe("var(--red)");
  });

  it("returns green when the balance increased", () => {
    expect(deltaColor(500, 1000)).toBe("var(--green)");
  });

  it("returns muted text color when the balance is unchanged", () => {
    expect(deltaColor(500, 500)).toBe("var(--text-muted)");
  });

  it("treats a full drain to zero as a decrease", () => {
    expect(deltaColor(1_980_800.05, 0)).toBe("var(--red)");
  });
});
