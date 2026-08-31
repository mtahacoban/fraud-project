import { beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_ORDER, STORAGE_KEY, loadOrder, saveOrder } from "./KpiCards.jsx";

function createMemoryStorage() {
  let store = {};
  return {
    getItem: (key) => (key in store ? store[key] : null),
    setItem: (key, value) => { store[key] = String(value); },
    removeItem: (key) => { delete store[key]; },
    clear: () => { store = {}; },
  };
}

beforeEach(() => {
  vi.stubGlobal("localStorage", createMemoryStorage());
});

describe("loadOrder", () => {
  it("returns the default order when nothing is saved yet", () => {
    expect(loadOrder()).toEqual(DEFAULT_ORDER);
  });

  it("returns a previously saved order unchanged, if it's still valid", () => {
    const custom = ["pending_proposals", "scored_transactions", "open_cases", "high_risk"];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(custom));
    expect(loadOrder()).toEqual(custom);
  });

  it("appends a card missing from the saved order, rather than dropping it", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(["open_cases", "scored_transactions", "pending_proposals"]));
    expect(loadOrder()).toEqual(["open_cases", "scored_transactions", "pending_proposals", "high_risk"]);
  });

  it("drops keys from the saved order that no longer exist today", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(["open_cases", "some_removed_card", "scored_transactions", "pending_proposals"]));
    expect(loadOrder()).toEqual(["open_cases", "scored_transactions", "pending_proposals", "high_risk"]);
  });

  it("falls back to the default order when the saved value isn't an array", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ not: "an array" }));
    expect(loadOrder()).toEqual(DEFAULT_ORDER);
  });

  it("falls back to the default order when the saved value is malformed JSON", () => {
    localStorage.setItem(STORAGE_KEY, "{not valid json");
    expect(loadOrder()).toEqual(DEFAULT_ORDER);
  });
});

describe("saveOrder", () => {
  it("persists an order that loadOrder can read back unchanged", () => {
    const custom = ["open_cases", "pending_proposals", "scored_transactions", "high_risk"];
    saveOrder(custom);
    expect(loadOrder()).toEqual(custom);
  });

  it("does not throw when localStorage.setItem fails (private browsing / quota)", () => {
    vi.stubGlobal("localStorage", {
      setItem: () => { throw new Error("QuotaExceededError"); },
      getItem: () => null,
    });
    expect(() => saveOrder(DEFAULT_ORDER)).not.toThrow();
  });
});
