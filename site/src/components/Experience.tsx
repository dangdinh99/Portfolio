import { SectionReveal } from './SectionReveal';

const items = [
  {
    date: 'Sep 2025 - Present',
    title: 'Master in Data Science',
    company: 'Boston University - Faculty of Computing and Data Sciences',
    icon: 'fa-graduation-cap',
    isSchool: true,
    description:
      'Focused on advanced machine learning, distributed systems, and real-time analytics. Coursework includes deep learning, cloud computing, and big data engineering.',
  },
  {
    date: 'Oct 2024 - May 2025',
    title: 'AI Researcher',
    company: 'Research Assistant, NC State University',
    icon: 'fa-briefcase',
    isSchool: false,
    description:
      'Led an in-depth study comparing general-purpose LLMs and RAG systems for education use cases—evaluating strengths, limitations, and when a pure LLM can substitute for RAG through the lenses of scalability, reliability, and accessibility.',
  },
  {
    date: 'May 2024 - May 2025',
    title: 'Network Operation Center TELE Intern',
    company: 'OIT ComTech (Raleigh, NC)',
    icon: 'fa-briefcase',
    isSchool: false,
    description:
      'Monitored network infrastructure and resolved technical issues. Collaborated with cross-functional teams to maintain system reliability and optimize performance.',
  },
  {
    date: 'Aug 2023 - May 2025',
    title: 'Bachelor of Science in Business Administration & Economics',
    company: 'North Carolina State University - Poole College of Management',
    icon: 'fa-graduation-cap',
    isSchool: true,
    description:
      'Dual degree in Business IT and Economics. Developed strong analytical and technical skills through coursework in data analytics, information systems, and econometrics.',
  },
  {
    date: 'Aug 2021 - May 2023',
    title: 'Associate in Arts Degree',
    company: 'Wake Technical Community College',
    icon: 'fa-graduation-cap',
    isSchool: true,
    description: "President's List. Built foundational knowledge in mathematics, computer science, and general education.",
  },
];

export function Experience() {
  return (
    <section id="experience">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Career</span>
          <h2 className="section-title">Experience & Education</h2>
        </div>

        <div className="timeline">
          {items.map((item) => (
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
