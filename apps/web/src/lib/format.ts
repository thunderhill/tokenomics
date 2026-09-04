export function usd(value: string | number, opts: Intl.NumberFormatOptions = {}): string {
  const n = typeof value === "string" ? Number(value) : value;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
    ...opts,
  }).format(n);
}

export function usdPrecise(value: string | number): string {
  return usd(value, { maximumFractionDigits: 6, minimumFractionDigits: 2 });
}

export function compactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", { notation: "compact" }).format(value);
}

export function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function shortDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function shortDateTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

/** Token counts run to the hundreds of millions; only the magnitude is readable. */
export function tokens(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

/**
 * Providers quote every price per million tokens, so effective rates are shown that
 * way too -- it is the one figure you can hold against a price list without arithmetic.
 */
export function usdPerM(value: string | number | null): string {
  if (value === null) return "-";
  const n = typeof value === "string" ? Number(value) : value;
  return `${usd(n, { maximumFractionDigits: n < 1 ? 3 : 2 })}/M`;
}

/** Signed money, for figures where the direction is the whole point (cache benefit). */
export function usdSigned(value: string | number): string {
  const n = typeof value === "string" ? Number(value) : value;
  return `${n > 0 ? "+" : ""}${usd(n)}`;
}
