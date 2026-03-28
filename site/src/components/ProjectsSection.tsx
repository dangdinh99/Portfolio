import { useState } from 'react';
import { projects } from '../data/projects';
import type { Project } from '../data/projectTypes';
import { ProjectModal } from './ProjectModal';
import { SectionReveal } from './SectionReveal';

function assetUrl(path: string) {
  return path.startsWith('/') ? path : `/${path}`;
}

export function ProjectsSection() {
  const [selected, setSelected] = useState<Project | null>(null);

  return (
    <>
      <section id="projects">
        <div className="container">
          <div className="section-header">
            <span className="section-tag">Portfolio</span>
            <h2 className="section-title">Featured Projects</h2>
            <p className="section-description">
              A collection of projects showcasing my skills in machine learning, data engineering, and full-stack
              development.
            </p>
          </div>

          <div className="projects-grid">
            {projects.map((p) => (
              <SectionReveal key={p.id} className="project-card">
                <img src={assetUrl(p.image.trim())} alt="" className="project-image" />
                <div className="project-content">
                  <span className="project-category">{p.category}</span>
                  <h3>{p.title}</h3>
                  <p className="project-description">{p.description}</p>
                  <div className="project-tech">
                    {p.tech.slice(0, 4).map((t) => (
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
                      onClick={() => setSelected(p)}
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
      <ProjectModal project={selected} onClose={() => setSelected(null)} />
    </>
  );
}
