# DESIGN.md — Pickleball.Vision

## Theme
**Athletic Editorial** — neo-brutalist vintage sports almanac. Warm printed paper,
ink-black structure, hard offset shadows, grain texture, two punchy accents.
Light theme. Brand surfaces loud; product surfaces restrained.

## Color palette (OKLCH)
| Token | Hex | OKLCH | Use |
|-------|-----|-------|-----|
| paper | `#f3efe3` | oklch(0.95 0.018 95) | page background |
| paper-2 | `#e9e2cf` | oklch(0.91 0.028 95) | raised surface / table head |
| ink | `#17150f` | oklch(0.20 0.012 80) | text, borders, shadows |
| ink-soft | `#3a362c` | oklch(0.36 0.015 90) | secondary text |
| lime | `#c2f000` | oklch(0.92 0.22 118) | primary action / highlight |
| flame | `#ea4b2a` | oklch(0.64 0.20 35) | labels, alerts, errors |

Contrast: ink-on-paper ≈ 14:1. lime/flame are **accent fills behind ink text** or
**bold ≥14px labels** only — never small body text on paper (flame-on-paper ≈ 3.3:1).

## Typography
- **Display:** Anton (poster condensed) — headings only, `text-wrap: balance`,
  letter-spacing ≥ -0.03em. Hero clamp ceiling **6rem**.
- **Editorial:** Fraunces (italic serif) — pull-quotes / lede only.
- **Body:** Spline Sans — 400–600, body line-length ≤ 70ch.
- **Mono:** JetBrains Mono — eyebrows, data, coords, logs.
- Max 4 families (display+serif+body+mono); scale ratio ≥ 1.25.
- Uppercase reserved for ≤4-word labels, eyebrows, badges. No all-caps sentences.

## Components
- **card** — `border-2 border-ink` + 4px hard offset shadow; hover nudges up-left.
  Use sparingly in the app (cards are not the default container).
- **btn-primary** — lime fill, ink border, hard shadow, press = shadow collapse.
- **btn-ghost** — paper fill, ink border, hard shadow.
- **input** — ink border; focus = lime offset shadow + visible ring.
- **badge / eyebrow** — mono uppercase micro-labels.
- **terminal** (admin logs) — inverted: ink bg, paper text, level-colored lines.

## Layout
- Asymmetric editorial grids on brand; calm 1-column/flex on product.
- Generous rhythm; hard dividers (2px ink) instead of soft shadows for sections.
- Responsive grids: `repeat(auto-fit, minmax(280px, 1fr))`.
- z-index scale: dropdown < sticky-nav < modal < toast < tooltip.

## Motion
- Entrances: `fadeup` staggered; ease-out (no bounce/elastic).
- Ambient: marquee strip, floating ball — **brand only**.
- **prefers-reduced-motion: reduce** → disable marquee/floaty, replace fadeup with
  instant/opacity. Never block content on motion.
