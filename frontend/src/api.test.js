import { describe, expect, it, vi } from "vitest";
import { exportCasesCsvUrl, exportCasesXlsxUrl, getCases } from "./api.js";

describe("exportCasesCsvUrl", () => {
  it("always sets format=csv regardless of what's passed in", () => {
    const url = exportCasesCsvUrl({ status: "OPEN" });
    expect(url).toContain("format=csv");
  });

  it("drops undefined/null/empty-string params, keeps real values", () => {
    const url = exportCasesCsvUrl({
      status: "OPEN", risk_band: undefined, q: null, country: "",
      type: "TRANSFER",
    });
    const qs = new URL(url, "http://x").searchParams;
    expect(qs.get("status")).toBe("OPEN");
    expect(qs.get("type")).toBe("TRANSFER");
    expect(qs.has("risk_band")).toBe(false);
    expect(qs.has("q")).toBe(false);
    expect(qs.has("country")).toBe(false);
  });

  it("produces a valid URL with no params at all beyond format", () => {
    const url = exportCasesCsvUrl();
    const qs = new URL(url, "http://x").searchParams;
    expect(qs.get("format")).toBe("csv");
    expect([...qs.keys()]).toEqual(["format"]);
  });

  it("passes all 9 filter params through untouched", () => {
    const params = {
      status: "OPEN", risk_band: "RED", q: "42", type: "TRANSFER",
      country: "TR", date_from: "2026-01-01", date_to: "2026-01-31",
      amount_min: "1000", amount_max: "5000",
    };
    const qs = new URL(exportCasesCsvUrl(params), "http://x").searchParams;
    for (const [key, value] of Object.entries(params)) {
      expect(qs.get(key)).toBe(value);
    }
  });
});

describe("exportCasesXlsxUrl", () => {
  it("always sets format=xlsx, and is otherwise identical to the CSV builder", () => {
    const params = { status: "CLOSED", country: "DE" };
    const csvQs = new URL(exportCasesCsvUrl(params), "http://x").searchParams;
    const xlsxQs = new URL(exportCasesXlsxUrl(params), "http://x").searchParams;
    expect(xlsxQs.get("format")).toBe("xlsx");
    expect(csvQs.get("format")).toBe("csv");
    expect(xlsxQs.get("status")).toBe(csvQs.get("status"));
    expect(xlsxQs.get("country")).toBe(csvQs.get("country"));
  });

  it("drops empty-string params the same way the CSV builder does", () => {
    const url = exportCasesXlsxUrl({ q: "" });
    const qs = new URL(url, "http://x").searchParams;
    expect(qs.has("q")).toBe(false);
  });
});

describe("getCases", () => {
  it("builds a query string from the params object, dropping empty values, and requests it", async () => {
    const calls = [];
    vi.stubGlobal("fetch", (url) => {
      calls.push(String(url));
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ items: [], total: 0 }) });
    });

    await getCases({ status: "OPEN", risk_band: "", q: undefined, offset: 50 });

    expect(calls).toHaveLength(1);
    const requestedUrl = new URL(calls[0], "http://x");
    expect(requestedUrl.pathname).toBe("/cases");
    expect(requestedUrl.searchParams.get("status")).toBe("OPEN");
    expect(requestedUrl.searchParams.get("offset")).toBe("50");
    expect(requestedUrl.searchParams.has("risk_band")).toBe(false);
    expect(requestedUrl.searchParams.has("q")).toBe(false);

    vi.unstubAllGlobals();
  });

  it("requests a bare /cases with no query string when called with no params", async () => {
    const calls = [];
    vi.stubGlobal("fetch", (url) => {
      calls.push(String(url));
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ items: [], total: 0 }) });
    });

    await getCases();

    expect(calls[0].endsWith("/cases")).toBe(true);
    vi.unstubAllGlobals();
  });
});
