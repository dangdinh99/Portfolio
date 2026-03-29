import { useState } from 'react';
import { projects } from '../data/projects';
import { contests } from '../data/contests';
import type { Project, Contest } from '../data/projectTypes';
import { ProjectModal } from './ProjectModal';
import { ContestModal } from './ContestModal';
import { SectionReveal } from './SectionReveal';

function assetUrl(path: string) {
  return path.startsWith('/') ? path : `/${path}`;
}

const FILTERS = ['All', 'ML', 'Data Engineering', 'AI / Ethics', 'Full Stack'];

function matchesFilter(project: Project, filter: string): boolean {
  if (filter === 'All') return true;
  const cat = project.category.toLowerCase();
  if (filter === 'ML')               return cat.includes('machine learning') || cat.includes('deep learning');
  if (filter === 'Data Engineering') return cat.includes('data engineer') || cat.includes('data pipeline');
  if (filter === 'AI / Ethics')      return cat.includes('ai ethic') || cat.includes('benchmark');
  if (filter === 'Full Stack')       return cat.includes('system') || cat.includes('automation') || cat.includes('full stack');
  return false;
}

export function ProjectsSection() {
  const [activeTab, setActiveTab] = useState<'projects' | 'competitions'>('projects');
  const [activeFilter, setActiveFilter] = useState('All');
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [selectedContest, setSelectedContest] = useState<Contest | null>(null);

  function switchTab(tab: 'projects' | 'competitions') {
    setActiveTab(tab);
    setActiveFilter('All');
  }

  const filteredProjects = projects.filter((p) => matchesFilter(p, activeFilter));

  return (
    <>
      <section id="projects">
        <div className="container">
          <div className="section-header">
            <span className="section-tag">Portfolio</span>
            <h2 className="section-title">
              {activeTab === 'projects' ? 'Featured Projects' : 'Contests & Highlights'}
            </h2>
            <p className="section-description">
              {activeTab === 'projects'
                ? 'A collection of projects showcasing my skills in machine learning, data engineering, and full-stack development.'
                : 'Selected competitions, collaborations, and standout academic or team challenges.'}
            </p>
          </div>

          {/* Tab switcher */}
          <div className="section-tabs">
            <button
              type="button"
              className={`tab ${activeTab === 'projects' ? 'active' : ''}`}
              onClick={() => switchTab('projects')}
            >
              <i className="fas fa-folder" /> Projects
            </button>
            <button
              type="button"
              className={`tab ${activeTab === 'competitions' ? 'active' : ''}`}
              onClick={() => switchTab('competitions')}
            >
              <i className="fas fa-trophy" /> Competitions
            </button>
          </div>

          {/* Filter bar — only on Projects tab */}
          {activeTab === 'projects' && (
            <div className="filter-bar">
              {FILTERS.map((f) => (
                <button
                  key={f}
                  type="button"
                  className={`filter-btn ${activeFilter === f ? 'active' : ''}`}
                  onClick={() => setActiveFilter(f)}
                >
                  {f}
                </button>
              ))}
            </div>
          )}

          {/* Grid */}
          <div className="projects-grid">
            {activeTab === 'projects'
              ? filteredProjects.map((p) => (
                  <SectionReveal key={p.id} className="project-card">
                    <img src={assetUrl(p.image.trim())} alt="" className="project-image" />
                    <div className="project-content">
                      <span className="project-category">{p.category}</span>
                      <h3>{p.title}</h3>
                      <p className="project-description">{p.description}</p>
                      <div className="project-tech">
                        {p.tech.slice(0, 4).map((t) => (
                          <span key={t} className="tech-badge">{t}</span>
                        ))}
                      </div>
                      <div className="project-links">
                        <button
                          type="button"
                          className="project-link"
                          style={{ background: 'none', border: 'none', padding: 0, font: 'inherit' }}
                          onClick={() => setSelectedProject(p)}
                        >
                          <i className="fas fa-book-open" /> Learn More
                        </button>
                      </div>
                    </div>
                  </SectionReveal>
                ))
              : contests.map((c) => (
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
                          <span key={t} className="tech-badge">{t}</span>
                        ))}
                      </div>
                      <div className="project-links">
                        <button
                          type="button"
                          className="project-link"
                          style={{ background: 'none', border: 'none', padding: 0, font: 'inherit' }}
                          onClick={() => setSelectedContest(c)}
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

      <ProjectModal project={selectedProject} onClose={() => setSelectedProject(null)} />
      <ContestModal contest={selectedContest} onClose={() => setSelectedContest(null)} />
    </>
  );
}
