// Fixed categorical order (never cycled): colors follow the entity, so a project
// keeps its color across every chart on the page for as long as the app runs.
//
// SVG presentation attributes (the `stroke`/`fill` props Recharts renders) do not
// reliably resolve `var(--…)` custom properties the way inline `style` does, so charts
// need real hex. CSS keeps using the custom properties directly; this module is the
// one place the two are kept in sync with `styles.css`.
const LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"] as const;
const DARK = ["#3987e5", "#d95926", "#199e70"] as const;
const FALLBACK = "#898781";

const prefersDark =
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-color-scheme: dark)").matches;

const SLOTS = prefersDark ? DARK : LIGHT;

const assigned = new Map<string, string>();

export function colorFor(key: string): string {
  const existing = assigned.get(key);
  if (existing) return existing;
  if (assigned.size >= SLOTS.length) return FALLBACK;
  const color = SLOTS[assigned.size]!;
  assigned.set(key, color);
  return color;
}

// The five billable components are a closed, ordered set, so they get fixed colors
// rather than slots from the cycling palette above: the input side runs cool (cache
// reads cheapest) and the output side warm, which is also the order they cost money in.
const COMPONENT_LIGHT: Record<string, string> = {
  cache_read: "#7fb2e8",
  cache_write: "#2a78d6",
  input: "#1baf7a",
  output: "#eb6834",
  reasoning: "#b3479c",
};

const COMPONENT_DARK: Record<string, string> = {
  cache_read: "#6ba3de",
  cache_write: "#3987e5",
  input: "#199e70",
  output: "#d95926",
  reasoning: "#c25aab",
};

export function componentColor(component: string): string {
  const palette = prefersDark ? COMPONENT_DARK : COMPONENT_LIGHT;
  return palette[component] ?? FALLBACK;
}
