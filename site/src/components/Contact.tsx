import { SectionReveal } from './SectionReveal';

export function Contact() {
  return (
    <section id="contact">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Get In Touch</span>
          <h2 className="section-title">Let&apos;s Connect</h2>
          <p className="section-description">
            Open to collaboration, opportunities, or just a chat about data science and technology.
          </p>
        </div>

        <div className="contact-grid">
          <SectionReveal className="contact-item">
            <div className="contact-icon">
              <i className="fas fa-envelope" />
            </div>
            <h3>Email</h3>
            <a href="mailto:dangdinh030@gmail.com">dangdinh030@gmail.com</a>
          </SectionReveal>

          <SectionReveal className="contact-item">
            <div className="contact-icon">
              <i className="fas fa-map-marker-alt" />
            </div>
            <h3>Location</h3>
            <p>Boston, MA 02135</p>
          </SectionReveal>

          <SectionReveal className="contact-item">
            <div className="contact-icon">
              <i className="fas fa-phone-alt" />
            </div>
            <h3>Phone</h3>
            <a href="tel:+19197462989">+1 (919) 746-2989</a>
          </SectionReveal>
        </div>

        <div className="social-links">
          <a href="https://www.linkedin.com/in/dang-dinh/" target="_blank" rel="noreferrer" className="social-link" aria-label="LinkedIn">
            <i className="fab fa-linkedin-in" />
          </a>
          <a href="https://github.com/dangdinh99" target="_blank" rel="noreferrer" className="social-link" aria-label="GitHub">
            <i className="fab fa-github" />
          </a>
          <a href="https://www.facebook.com/thai.duy.1466/" target="_blank" rel="noreferrer" className="social-link" aria-label="Facebook">
            <i className="fab fa-facebook-f" />
          </a>
          <a href="https://www.instagram.com/_gnad_02/" target="_blank" rel="noreferrer" className="social-link" aria-label="Instagram">
            <i className="fab fa-instagram" />
          </a>
        </div>
      </div>
    </section>
  );
}
