import { useState } from 'react';
import { contests } from '../data/contests';
import type { Contest } from '../data/projectTypes';
import { ContestModal } from './ContestModal';
import { SectionReveal } from './SectionReveal';

function assetUrl(path: string) {
  return path.startsWith('/') ? path : `/${path}`;
}

export function ContestsSection() {
  const [selected, setSelected] = useState<Contest | null>(null);

  return (
    <>
      <section id="contests">
        <div className="container">
          <div className="section-header">
            <span className="section-tag">Competitions</span>
            <h2 className="section-title">Contests & Highlights</h2>
            <p className="section-description">
              Selected competitions, collaborations, and standout academic or team challenges.
            </p>
          </div>

          <div className="contests-grid">
            {contests.map((c) => (
              <SectionReveal key={c.id} className="project-card">
                {c.image ? (
                  <img src={assetUrl(c.image.trim())} alt="" className="project-image" />
                ) : (
                  <div
                    className="project-image"
                    style={{
                      background: 'var(--bg-card)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--accent)',
                    }}
                  >
                    <i className="fas fa-trophy fa-3x" aria-hidden />
                  </div>
                )}
                <div className="project-content">
                  <span className="project-category">{c.organizer}</span>
                  <h3>{c.title}</h3>
                  <p className="contest-result">{c.result}</p>
                  <p className="project-description">{c.description}</p>
                  <div className="project-tech">
                    {c.tech.slice(0, 4).map((t) => (
                      <span key={t} className="tech-badge">
                        {t}
                      </span>
                    ))}
                  </div>
                  <div className="project-links">
                    <button
                      type="button"
                      className="project-link"
                      style={{ background: 'none', border: 'none', padding: 0, font: 'inherit' }}
                      onClick={() => setSelected(c)}
                    >
                      <i className="fas fa-book-open" /> Learn More
                    </button>
                  </div>
                </div>
              </SectionReveal>
            ))}
          </div>
        </div>
      </section>
      <ContestModal contest={selected} onClose={() => setSelected(null)} />
    </>
  );
}
