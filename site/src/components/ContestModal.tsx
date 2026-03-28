import { useEffect } from 'react';
import type { Contest } from '../data/projectTypes';

function assetUrl(path: string) {
  return path.startsWith('/') ? path : `/${path}`;
}

type Props = {
  contest: Contest | null;
  onClose: () => void;
};

export function ContestModal({ contest, onClose }: Props) {
  useEffect(() => {
    if (!contest) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.body.classList.add('modal-open');
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.classList.remove('modal-open');
      window.removeEventListener('keydown', onKey);
    };
  }, [contest, onClose]);

  if (!contest) return null;

  const bodyHtml = contest.longDescription ?? `<p>${contest.description}</p>`;

  return (
    <div
      className="modal-overlay active"
      role="dialog"
      aria-modal="true"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal-content">
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
          <i className="fas fa-times" />
        </button>

        <div className="modal-header">
          {contest.image ? (
            <img src={assetUrl(contest.image.trim())} alt="" className="modal-header-image" />
          ) : null}
          <div className="modal-header-overlay">
            <span className="modal-category">{contest.organizer}</span>
            <h2 className="modal-title">{contest.title}</h2>
          </div>
        </div>

        <div className="modal-body">
          <div className="modal-section">
            <p className="modal-description" style={{ marginBottom: '0.5rem' }}>
              <strong>{contest.date}</strong> — <span className="text-success">{contest.result}</span>
            </p>
            <div className="modal-description" dangerouslySetInnerHTML={{ __html: bodyHtml }} />
          </div>
          <div className="modal-section">
            <h3>
              <i className="fas fa-tools" /> Technologies
            </h3>
            <div className="modal-tech-grid">
              {contest.tech.map((t) => (
                <span key={t} className="modal-tech-badge">
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="modal-actions">
          {contest.link ? (
            <a href={contest.link} target="_blank" rel="noreferrer" className="modal-btn modal-btn-primary">
              <i className="fas fa-external-link-alt" /> Open link
            </a>
          ) : null}
          <button type="button" className="modal-btn modal-btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
