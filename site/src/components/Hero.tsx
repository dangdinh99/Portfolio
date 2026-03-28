function assetUrl(path: string) {
  return path.startsWith('/') ? path : `/${path}`;
}

export function Hero() {
  return (
    <section id="hero">
      <div className="container">
        <div className="hero-container">
          <div className="hero-left">
            <span className="hero-greeting">Hello, I&apos;m Dang Dinh</span>
            <h1 className="hero-title">
              Building intelligent systems with <span>data & code</span>
            </h1>
            <p className="hero-tagline">MS in Data Science @ Boston University</p>
            <p className="hero-description">
              I specialize in machine learning, real-time analytics, and data-driven solutions. From building
              predictive models to deploying scalable pipelines, I turn complex data into actionable insights.
            </p>
            <div className="hero-cta">
              <a href="#projects" className="btn btn-primary">
                View Projects <i className="fas fa-arrow-right" />
              </a>
              <a href={assetUrl('files/DangDinh_ResumeDS.pdf')} download className="btn btn-secondary">
                <i className="fas fa-download" /> Resume
              </a>
            </div>
          </div>
          <div className="hero-right">
            <div className="profile-image-wrapper">
              <img src={assetUrl('image/about1.png')} alt="Dang Dinh" className="profile-pic" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
