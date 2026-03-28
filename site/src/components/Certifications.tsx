import { SectionReveal } from './SectionReveal';

const certs = [
  { iconClass: 'fas fa-snowflake', title: 'Snowflake Certified', provider: 'Coursera', date: '2025', badge: 'Data Warehouse' },
  { iconClass: 'fab fa-python', title: 'Python for Data Science', provider: 'Codecademy', date: '2025', badge: 'Programming' },
  { iconClass: 'fab fa-aws', title: 'AWS Cloud Practitioner', provider: 'Amazon Web Services', date: ' Winter 2025', badge: 'Cloud Computing' },
];

export function Certifications() {
  return (
    <section id="certifications">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Credentials</span>
          <h2 className="section-title">Certifications</h2>
          <p className="section-description">
            Professional certifications validating my expertise in data science and cloud technologies.
          </p>
        </div>

        <div className="cert-grid">
          {certs.map((c) => (
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
    </section>
  );
}
