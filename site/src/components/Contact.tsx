import { useState } from 'react';
import emailjs from '@emailjs/browser';
import { SectionReveal } from './SectionReveal';
import { EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, EMAILJS_PUBLIC_KEY } from '../data/emailConfig';

function ContactForm() {
  const [form, setForm] = useState({ name: '', email: '', message: '' });
  const [status, setStatus] = useState<'idle' | 'sending' | 'success' | 'error'>('idle');

  function handleChange(e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus('sending');
    try {
      await emailjs.send(
        EMAILJS_SERVICE_ID,
        EMAILJS_TEMPLATE_ID,
        { from_name: form.name, from_email: form.email, message: form.message },
        EMAILJS_PUBLIC_KEY,
      );
      setStatus('success');
      setForm({ name: '', email: '', message: '' });
    } catch {
      setStatus('error');
    }
  }

  return (
    <form className="contact-form" onSubmit={handleSubmit}>
      <h3>Send a Message</h3>
      <div className="form-group">
        <label htmlFor="cf-name">Name</label>
        <input
          id="cf-name"
          name="name"
          type="text"
          value={form.name}
          onChange={handleChange}
          required
          placeholder="Your name"
        />
      </div>
      <div className="form-group">
        <label htmlFor="cf-email">Email</label>
        <input
          id="cf-email"
          name="email"
          type="email"
          value={form.email}
          onChange={handleChange}
          required
          placeholder="you@example.com"
        />
      </div>
      <div className="form-group">
        <label htmlFor="cf-message">Message</label>
        <textarea
          id="cf-message"
          name="message"
          rows={6}
          value={form.message}
          onChange={handleChange}
          required
          placeholder="What's on your mind?"
        />
      </div>
      <button type="submit" className="btn btn-primary contact-submit" disabled={status === 'sending'}>
        {status === 'sending' ? (
          <><i className="fas fa-spinner fa-spin" /> Sending…</>
        ) : (
          <><i className="fas fa-paper-plane" /> Send Message</>
        )}
      </button>
      {status === 'success' && (
        <p className="form-feedback form-feedback--success">
          <i className="fas fa-check-circle" /> Message sent! I&apos;ll get back to you soon.
        </p>
      )}
      {status === 'error' && (
        <p className="form-feedback form-feedback--error">
          <i className="fas fa-exclamation-circle" /> Something went wrong. Please email me directly.
        </p>
      )}
    </form>
  );
}

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

        <div className="contact-layout">
          <div className="contact-info-col">
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

          <SectionReveal className="contact-form-col">
            <ContactForm />
          </SectionReveal>
        </div>
      </div>
    </section>
  );
}
