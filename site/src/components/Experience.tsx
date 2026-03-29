import { SectionReveal } from './SectionReveal';
import { experience } from '../data/experience';

export function Experience() {
  return (
    <section id="experience">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Career</span>
          <h2 className="section-title">Experience & Education</h2>
        </div>

        <div className="timeline">
          {experience.map((item) => (
            <SectionReveal key={item.title + item.date} className="timeline-item">
              <div className="timeline-content">
                <span className="timeline-date">{item.date}</span>
                <h3>{item.title}</h3>
                <div className="timeline-company">
                  <i className={`fas ${item.icon}`} />
                  {item.company}
                </div>
                <p className="timeline-description">{item.description}</p>
              </div>
            </SectionReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
