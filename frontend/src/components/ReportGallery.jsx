/** @file Shared chart gallery for chat and live Ridge forecasts. */

import { Images } from 'lucide-react';

/** Render browser-safe chart URLs supplied by the backend artifact API. */
export default function ReportGallery({ attachments }) {
  if (!attachments?.length) return null;

  return (
    <section className="report-attachments" aria-label="模型評估圖表">
      <h3><Images size={17} />模型評估圖表</h3>
      <div className="report-gallery">
        {attachments.map((attachment) => (
          <figure className="report-chart" key={attachment.url}>
            <a
              href={attachment.url}
              target="_blank"
              rel="noreferrer"
              aria-label={`開啟${attachment.title}完整圖表`}
            >
              <img
                src={attachment.url}
                alt={attachment.alt || attachment.title}
                loading="lazy"
              />
            </a>
            <figcaption>{attachment.title}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}
