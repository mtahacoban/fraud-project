import { describe, expect, it } from "vitest";
import { automationGateValueText, automationGateVisualState, countryFlagEmoji } from "./CaseDetailPanel.jsx";

describe("countryFlagEmoji", () => {
  it("converts a 2-letter ISO code to its regional-indicator flag emoji", () => {
    expect(countryFlagEmoji("TR")).toBe("🇹🇷");
    expect(countryFlagEmoji("US")).toBe("🇺🇸");
  });

  it("is case-insensitive", () => {
    expect(countryFlagEmoji("tr")).toBe(countryFlagEmoji("TR"));
  });

  it("returns null for missing or malformed input", () => {
    expect(countryFlagEmoji(null)).toBeNull();
    expect(countryFlagEmoji(undefined)).toBeNull();
    expect(countryFlagEmoji("")).toBeNull();
    expect(countryFlagEmoji("USA")).toBeNull();
    expect(countryFlagEmoji("U")).toBeNull();
  });
});

describe("automationGateVisualState", () => {
  it("returns 'pass' for any passed gate, regardless of actual/threshold", () => {
    expect(automationGateVisualState({ passed: true, actual: 0.5, threshold: 0.95, gate: "similarity" })).toBe("pass");
    expect(automationGateVisualState({ passed: true, actual: null, threshold: null, gate: "hard_rule_conflict" })).toBe("pass");
  });

  it("returns 'fail' (not 'close') for a failed boolean-only gate", () => {
    expect(automationGateVisualState({ passed: false, actual: null, threshold: null, gate: "direction_automatable" })).toBe("fail");
  });

  it("returns 'close' for a ratio gate that missed by less than 0.05", () => {
    expect(automationGateVisualState({ passed: false, actual: 0.91, threshold: 0.95, gate: "similarity" })).toBe("close");
  });

  it("returns 'fail' for a ratio gate that missed by 0.05 or more", () => {
    expect(automationGateVisualState({ passed: false, actual: 0.50, threshold: 0.95, gate: "similarity" })).toBe("fail");
  });

  it("uses an absolute gap of 2 (not the 0.05 ratio) for precedent_count", () => {
    expect(automationGateVisualState({ passed: false, actual: 8, threshold: 10, gate: "precedent_count" })).toBe("close");
    expect(automationGateVisualState({ passed: false, actual: 3, threshold: 10, gate: "precedent_count" })).toBe("fail");
  });

  it("treats gap===2 as 'close' for precedent_count (isClose uses <=, not <)", () => {
    expect(automationGateVisualState({ passed: false, actual: 8, threshold: 10, gate: "precedent_count" })).toBe("close");
    expect(automationGateVisualState({ passed: false, actual: 7, threshold: 10, gate: "precedent_count" })).toBe("fail");
  });
});

describe("automationGateValueText", () => {
  it("formats a ratio gate as a percentage with its threshold", () => {
    expect(automationGateValueText({ actual: 0.9945, threshold: 0.95, gate: "similarity" })).toBe("99.5% (need ≥ 95%)");
  });

  it("formats precedent_count as a plain integer comparison, not a percentage", () => {
    expect(automationGateValueText({ actual: 12, threshold: 10, gate: "precedent_count" })).toBe("12 (need ≥ 10)");
  });

  it("falls back to the raw detail string for boolean-only gates", () => {
    const detail = "direction=fraud - automatable under current policy";
    expect(automationGateValueText({ actual: null, threshold: null, gate: "direction_automatable", detail })).toBe(detail);
  });
});
