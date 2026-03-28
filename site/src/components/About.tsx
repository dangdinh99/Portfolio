import { SectionReveal } from './SectionReveal';

const programming = ['Python', 'R', 'SQL', 'JavaScript', 'HTML/CSS'];
const ml = ['Scikit-learn', 'XGBoost', 'PyTorch', 'Pandas', 'NumPy', 'Matplotlib'];
const tools = ['AWS', 'Snowflake', 'Git', 'Docker', 'Tableau', 'Jupyter', 'VS Code', 'MySQL', 'PostgreSQL', 'Stata', 'Excel'];
const expertise = [
  'Machine Learning ',
  'Deep learning model',
  'Exploratory data analysis',
  'Data engineering',
  'Data cleaning',
  'NLP',
  'Big Data',
];

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
              <SectionReveal className="stat-item">
                <div className="stat-number">6+</div>
                <div className="stat-label">Projects Completed</div>
              </SectionReveal>
              <SectionReveal className="stat-item">
                <div className="stat-number">3.5+</div>
                <div className="stat-label">Years Learning</div>
              </SectionReveal>
              <SectionReveal className="stat-item">
                <div className="stat-number">2</div>
                <div className="stat-label">Certifications</div>
              </SectionReveal>
              <SectionReveal className="stat-item">
                <div className="stat-number">10+</div>
                <div className="stat-label">Technologies</div>
              </SectionReveal>
            </div>
          </div>

          <div className="skills-section">
            <SkillCategory icon="fa-code" title="Programming Languages" tags={programming} />
            <SkillCategory icon="fa-brain" title="ML & Data Science" tags={ml} />
            <SkillCategory icon="fa-tools" title="Tools & Platforms" tags={tools} />
            <SkillCategory icon="fa-graduation-cap" title="Expertise" tags={expertise} />
          </div>
        </div>
      </div>
    </section>
  );
}

function SkillCategory({ icon, title, tags }: { icon: string; title: string; tags: string[] }) {
  return (
    <SectionReveal className="skill-category">
      <h3>
        <i className={`fas ${icon}`} /> {title}
      </h3>
      <div className="skill-tags">
        {tags.map((t) => (
          <span key={t} className="skill-tag">
            {t}
          </span>
        ))}
      </div>
    </SectionReveal>
  );
}
