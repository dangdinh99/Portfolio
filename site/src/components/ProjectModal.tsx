import { useEffect } from 'react';
import type { Project } from '../data/projectTypes';

function assetUrl(path: string) {
  return path.startsWith('/') ? path : `/${path}`;
}

type Props = {
  project: Project | null;
  onClose: () => void;
};

export function ProjectModal({ project, onClose }: Props) {
  useEffect(() => {
    if (!project) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.body.classList.add('modal-open');
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.classList.remove('modal-open');
      window.removeEventListener('keydown', onKey);
    };
  }, [project, onClose]);

  if (!project) return null;

  const showStats = project.stats && project.stats.length > 0;

  return (
    <div
      className={`modal-overlay active`}
      id="projectModal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="modalTitleText"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal-content">
        <button type="button" className="modal-close" id="modalClose" onClick={onClose} aria-label="Close">
          <i className="fas fa-times" />
        </button>

        <div className="modal-header">
          <img
            src={assetUrl(project.image.trim())}
            alt={project.title}
            className="modal-header-image"
          />
          <div className="modal-header-overlay">
            <span className="modal-category">{project.category}</span>
            <h2 className="modal-title" id="modalTitleText">
              {project.title}
            </h2>
          </div>
        </div>

        <div className="modal-body">
          <div className="modal-section">
            <h3>
              <i className="fas fa-info-circle" /> Overview
            </h3>
            <div
              className="modal-description"
              dangerouslySetInnerHTML={{ __html: project.longDescription }}
            />
          </div>

          <div className="modal-section">
            <h3>
              <i className="fas fa-tools" /> Technologies Used
            </h3>
            <div className="modal-tech-grid">
              {project.tech.map((t) => (
                <span key={t} className="modal-tech-badge">
                  {t}
                </span>
              ))}
            </div>
          </div>

          <div className="modal-section">
            <h3>
              <i className="fas fa-star" /> Key Features & Achievements
            </h3>
            <ul className="modal-features-list">
              {project.features.map((f) => (
                <li key={f}>
                  <i className="fas fa-check-circle" /> {f}
                </li>
              ))}
            </ul>
          </div>

          {showStats ? (
            <div className="modal-section" id="modalStatsSection">
              <h3>
                <i className="fas fa-chart-line" /> Impact & Results
              </h3>
              <div className="modal-stats">
                {project.stats!.map((s) => (
                  <div key={s.label} className="modal-stat">
                    <div className="modal-stat-value">{s.value}</div>
                    <div className="modal-stat-label">{s.label}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>

        <div className="modal-actions" id="modalActions">
          {project.hasGithub && project.github ? (
            <a href={project.github} target="_blank" rel="noreferrer" className="modal-btn modal-btn-primary">
              <i className="fab fa-github" /> View on GitHub
            </a>
          ) : null}
          {project.hasDemo && project.demo ? (
            <a href={project.demo} target="_blank" rel="noreferrer" className="modal-btn modal-btn-secondary">
              <i className="fas fa-external-link-alt" /> Live Demo
            </a>
          ) : null}
          {project.hasReport && project.report ? (
            <a
              href={assetUrl(project.report)}
              target="_blank"
              rel="noreferrer"
              className="modal-btn modal-btn-secondary"
            >
              <i className="fas fa-file-pdf" /> View Report
            </a>
          ) : null}
          {!project.hasGithub && !project.hasDemo && !project.hasReport ? (
            <button type="button" className="modal-btn modal-btn-primary" onClick={onClose}>
              <i className="fas fa-times" /> Close
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
