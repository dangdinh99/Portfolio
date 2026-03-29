import { SectionReveal } from './SectionReveal';
import { skillGroups, expertiseTags } from '../data/skills';
import { certifications } from '../data/certifications';
import { useCountUp } from '../hooks/useCountUp';
import type { Skill, SkillGroup } from '../data/projectTypes';

const aboutStats = [
  { end: 6,  suffix: '+',   label: 'Projects Completed' },
  { end: 35, suffix: '',    label: 'Years Learning', format: (n: number) => `${(n / 10).toFixed(1)}+` },
  { end: 3,  suffix: '',    label: 'Certifications' },
  { end: 10, suffix: '+',   label: 'Technologies' },
];

function StatItem({ end, suffix, label, format }: typeof aboutStats[number]) {
  const { ref, count } = useCountUp(end, 1500);
  const display = format ? format(count) : `${count}${suffix}`;
  return (
    <SectionReveal className="stat-item">
      <div ref={ref as React.RefObject<HTMLDivElement>} className="stat-number">{display}</div>
      <div className="stat-label">{label}</div>
    </SectionReveal>
  );
}

function SkillBar({ skill }: { skill: Skill }) {
  const { ref, count } = useCountUp(skill.level, 900);
  return (
    <div className="skill-bar-row" ref={ref as React.RefObject<HTMLDivElement>}>
      <div className="skill-bar-header">
        <span className="skill-bar-name">{skill.name}</span>
        <span className="skill-bar-pct">{count}%</span>
      </div>
      <div className="skill-bar-track">
        <div className="skill-bar-fill" style={{ width: `${count}%` }} />
      </div>
    </div>
  );
}

function SkillBarGroup({ group }: { group: SkillGroup }) {
  return (
    <SectionReveal className="skill-category">
      <h3>
        <i className={`fas ${group.icon}`} /> {group.title}
      </h3>
      <div className="skill-bars">
        {group.skills.map((skill) => (
          <SkillBar key={skill.name} skill={skill} />
        ))}
      </div>
    </SectionReveal>
  );
}

export function About() {
  return (
    <section id="about">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">About Me</span>
          <h2 className="section-title">Who I Am</h2>
        </div>

        <div className="about-content">
          <div className="about-text">
            <p>
              I&apos;m a graduate student at Boston University pursuing a Master&apos;s in Data Science, with a
              foundation in Business IT and Economics from NC State. My approach combines technical expertise with
              business acumen to deliver solutions that drive real impact.
            </p>
            <p>
              Currently, I&apos;m expanding my skills in machine learning, distributed systems, and cloud
              architecture. I believe the best way to learn is by building—each project pushes me to solve real
              problems and explore new technologies.
            </p>
            <p>
              When I&apos;m not buried in data pipelines, I&apos;m either at the pool table chasing that perfect run,
              or at home playing video games to reset my brain after a long day of debugging and experimentation.
              Both keep me competitive, focused, and honestly help me come back to my work with a fresh mindset.
            </p>

            <div className="stats-grid">
              {aboutStats.map((s) => (
                <StatItem key={s.label} {...s} />
              ))}
            </div>
          </div>

          <div className="skills-section">
            {skillGroups.map((group) => (
              <SkillBarGroup key={group.title} group={group} />
            ))}
            <SectionReveal className="skill-category">
              <h3>
                <i className="fas fa-graduation-cap" /> Expertise
              </h3>
              <div className="skill-tags">
                {expertiseTags.map((t) => (
                  <span key={t} className="skill-tag">{t}</span>
                ))}
              </div>
            </SectionReveal>
          </div>
        </div>

        {/* Certifications sub-section */}
        <div className="certs-subsection">
          <h3 className="certs-subsection-title">
            <i className="fas fa-certificate" /> Certifications
          </h3>
          <div className="cert-grid">
            {certifications.map((c) => (
              <SectionReveal key={c.title} className="cert-card">
                <div className="cert-icon">
                  <i className={c.iconClass} />
                </div>
                <h3>{c.title}</h3>
                <p className="cert-provider">{c.provider}</p>
                <span className="cert-date">{c.date}</span>
                <span className="cert-badge">{c.badge}</span>
              </SectionReveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
