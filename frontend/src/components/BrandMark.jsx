/** @file Reusable visual identity mark. */

import { Activity } from 'lucide-react';

/** Render a decorative, size-configurable EdgeMind logo. */
export default function BrandMark({ size = 32 }) {
  return (
    <div
      className="brand-mark"
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <Activity size={size * 0.52} strokeWidth={2.3} />
    </div>
  );
}
