// === Navbar Scroll Effect ===
const navbar = document.getElementById('navbar');
const navLinks = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
  if (window.scrollY > 50) {
    navbar.classList.add('scrolled');
  } else {
    navbar.classList.remove('scrolled');
  }
  
  updateActiveNavLink();
});

// === Mobile Menu Toggle ===
const menuToggle = document.querySelector('.menu-toggle');
const navMenu = document.getElementById('nav-menu');

menuToggle.addEventListener('click', () => {
  navMenu.classList.toggle('show');
  document.body.classList.toggle('menu-open');
  
  if (navMenu.classList.contains('show')) {
    menuToggle.classList.remove('fa-bars');
    menuToggle.classList.add('fa-xmark');
  } else {
    menuToggle.classList.remove('fa-xmark');
    menuToggle.classList.add('fa-bars');
  }
});

navLinks.forEach(link => {
  link.addEventListener('click', () => {
    navMenu.classList.remove('show');
    document.body.classList.remove('menu-open');
    menuToggle.classList.remove('fa-xmark');
    menuToggle.classList.add('fa-bars');
  });
});

document.addEventListener('click', (e) => {
  if (navMenu.classList.contains('show') && 
      !navMenu.contains(e.target) && 
      !menuToggle.contains(e.target)) {
    navMenu.classList.remove('show');
    document.body.classList.remove('menu-open');
    menuToggle.classList.remove('fa-xmark');
    menuToggle.classList.add('fa-bars');
  }
});

// === Smooth Scroll with Active Nav Link ===
function updateActiveNavLink() {
  const sections = document.querySelectorAll('section[id]');
  const scrollY = window.pageYOffset;

  sections.forEach(section => {
    const sectionHeight = section.offsetHeight;
    const sectionTop = section.offsetTop - 100;
    const sectionId = section.getAttribute('id');
    const navLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);

    if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
      navLinks.forEach(link => link.classList.remove('active'));
      if (navLink) navLink.classList.add('active');
    }
  });
}

// === PROJECT MODAL SYSTEM ===

// Project data structure - YOU WILL FILL THIS IN
const projectsData = {
  'project1': {
    title: 'Differentiate Privacy on Voting Service (Fider)',
    category: 'Data Privacy',
    image: 'image/privacy.jpg',
    description: `This is a detailed description of your project. Explain what problem it solves, the approach you took, and the impact it had. Be specific about challenges you overcame and technologies you used.`,
    longDescription: `<strong><i>Problem:</i></strong><br>
Traditional voting systems expose exact vote counts in real-time, enabling observers to infer individual behavior through:
<br>• Timing attacks (watching counts change when someone votes)
<br>• Averaging attacks (querying repeatedly to remove noise)
<br>• Threshold crossing leaks (seeing when a count becomes visible)

<br><br><strong><i>Solution:</i></strong><br>
Built a differential privacy sidecar that:

<br>• Adds calibrated Laplace noise (ε=0.5) to vote counts
<br>• Releases results on fixed window timeframe (prevents timing attacks)
<br>• Reuses noise when counts unchanged (prevents averaging attacks)
<br>• Tracks privacy budget and lock the post when budget is exhaust.

<br><br><strong><i>Technical Architecture:</i></strong>
<br>• Frontend: Vanilla JavaScript displaying noisy counts with uncertainty ranges
<br>• Backend: FastAPI + PostgreSQL dual-database architecture
<br>• Privacy: ε-differential privacy with Laplace mechanism
<br>• Budget: Lifetime epsilon tracking with automatic post locking
<br>• Scheduler: APScheduler for batch publishing the noisy counts in a specific time`,
    tech: ['Python', 'FastAPI', 'Docker', 'SQL', 'Javascript', 'HTML/CSS'],
    features: [
      'Prevents timing attacks (fixed-schedule releases)',
      'Prevents averaging attacks (noise reuse)',
      'Budget tracking (lifetime epsilon cap)',
      'Post locking (final results after budget exhaustion)',
      'Individual votes protected with plausible deniability',
      'Balances utility with privacy'
    ],
    stats: [
      { value: '100%', label: 'Attack Prevention' },
      { value: '96%', label: 'Decision Accuracy' },
      { value: '98.3%', label: 'Privacy Budget Saving' }
    ],
    github: 'https://github.com/dangdinh99/fider-dp',
    demo: 'https://your-demo-link.com',
    hasGithub: true,
    hasDemo: true
  },
  'project2': {
    title: 'Campaign Zero: 911 Call Data',
    category: 'Data Pipeline | Data Engineer',
    image: 'image/campaignzero.png ',
    description: 'Another detailed project description...',
    longDescription: 'Extended description for project 2...',
    tech: ['Python', 'Snowflake', 'Airflow', 'SQL'],
    features: [
      'Feature #1',
      'Feature #2',
      'Feature #3',
      'Feature #4'
    ],
    stats: [
      { value: '60M+', label: '911 Calls data' },
      { value: '11 Cities', label: '1 Schema' },
      { value: 'Scalable', label: 'Pipeline' }
    ],
    github: 'https://github.com/BU-Spark/ds-cz-911/tree/fa25-team-a-dev',
    demo: null,
    hasGithub: true,
    hasDemo: false
  },
  'project3': {
    title: 'Project Title 3',
    category: 'Data Privacy',
    image: 'image/project-placeholder.jpg',
    description: 'Project 3 description...',
    longDescription: 'Extended description for project 3...',
    tech: ['Python', 'Encryption', 'Security'],
    features: [
      'Feature #1',
      'Feature #2',
      'Feature #3'
    ],
    stats: [
      { value: '256-bit', label: 'Encryption' },
      { value: '100%', label: 'Secure' },
      { value: '<1ms', label: 'Overhead' }
    ],
    github: 'https://github.com/yourusername/project3',
    demo: null,
    hasGithub: true,
    hasDemo: false
  },
  'project4': {
    title: 'Project Title 4',
    category: 'Analytics',
    image: 'image/project-placeholder.jpg',
    description: 'Project 4 description...',
    longDescription: 'Extended description for project 4...',
    tech: ['R', 'Tableau', 'SQL', 'Statistics'],
    features: [
      'Feature #1',
      'Feature #2',
      'Feature #3'
    ],
    stats: [
      { value: '15+', label: 'Insights' },
      { value: '40%', label: 'Improvement' },
      { value: '10K', label: 'Users' }
    ],
    github: null,
    demo: 'https://tableau-public.com/your-dashboard',
    hasGithub: false,
    hasDemo: true
  }
};

// Create modal HTML
function createModalHTML() {
  const modalHTML = `
    <div class="modal-overlay" id="projectModal">
      <div class="modal-content">
        <button class="modal-close" id="modalClose">
          <i class="fas fa-times"></i>
        </button>
        
        <div class="modal-header">
          <img src="" alt="" class="modal-header-image" id="modalImage">
          <div class="modal-header-overlay">
            <span class="modal-category" id="modalCategory"></span>
            <h2 class="modal-title" id="modalTitle"></h2>
          </div>
        </div>
        
        <div class="modal-body">
          <div class="modal-section">
            <h3><i class="fas fa-info-circle"></i> Overview</h3>
            <p class="modal-description" id="modalDescription"></p>
          </div>
          
          <div class="modal-section">
            <h3><i class="fas fa-tools"></i> Technologies Used</h3>
            <div class="modal-tech-grid" id="modalTech"></div>
          </div>
          
          <div class="modal-section">
            <h3><i class="fas fa-star"></i> Key Features & Achievements</h3>
            <ul class="modal-features-list" id="modalFeatures"></ul>
          </div>
          
          <div class="modal-section" id="modalStatsSection">
            <h3><i class="fas fa-chart-line"></i> Impact & Results</h3>
            <div class="modal-stats" id="modalStats"></div>
          </div>
        </div>
        
        <div class="modal-actions" id="modalActions"></div>
      </div>
    </div>
  `;
  
  document.body.insertAdjacentHTML('beforeend', modalHTML);
}

// Open modal with project data
function openProjectModal(projectId) {
  const project = projectsData[projectId];
  if (!project) return;
  
  const modal = document.getElementById('projectModal');
  
  // Populate modal content
  document.getElementById('modalImage').src = project.image;
  document.getElementById('modalImage').alt = project.title;
  document.getElementById('modalCategory').textContent = project.category;
  document.getElementById('modalTitle').textContent = project.title;
  document.getElementById('modalDescription').innerHTML = project.longDescription;
  
  // Populate tech stack
  const techContainer = document.getElementById('modalTech');
  techContainer.innerHTML = project.tech.map(tech => 
    `<span class="modal-tech-badge">${tech}</span>`
  ).join('');
  
  // Populate features
  const featuresContainer = document.getElementById('modalFeatures');
  featuresContainer.innerHTML = project.features.map(feature => 
    `<li><i class="fas fa-check-circle"></i> ${feature}</li>`
  ).join('');
  
  // Populate stats (if available)
  if (project.stats && project.stats.length > 0) {
    const statsContainer = document.getElementById('modalStats');
    statsContainer.innerHTML = project.stats.map(stat => 
      `<div class="modal-stat">
        <div class="modal-stat-value">${stat.value}</div>
        <div class="modal-stat-label">${stat.label}</div>
      </div>`
    ).join('');
  } else {
    document.getElementById('modalStatsSection').style.display = 'none';
  }
  
  // Populate action buttons
  const actionsContainer = document.getElementById('modalActions');
  let actionsHTML = '';
  
  if (project.hasGithub && project.github) {
    actionsHTML += `
      <a href="${project.github}" target="_blank" class="modal-btn modal-btn-primary">
        <i class="fab fa-github"></i> View on GitHub
      </a>
    `;
  }
  
  if (project.hasDemo && project.demo) {
    actionsHTML += `
      <a href="${project.demo}" target="_blank" class="modal-btn modal-btn-secondary">
        <i class="fas fa-external-link-alt"></i> ${project.hasGithub ? 'Live Demo' : 'View Project'}
      </a>
    `;
  }
  
  // If no links, add a close button
  if (!actionsHTML) {
    actionsHTML = `
      <button class="modal-btn modal-btn-primary" onclick="closeProjectModal()">
        <i class="fas fa-times"></i> Close
      </button>
    `;
  }
  
  actionsContainer.innerHTML = actionsHTML;
  
  // Show modal
  modal.classList.add('active');
  document.body.classList.add('modal-open');
}

// Close modal
function closeProjectModal() {
  const modal = document.getElementById('projectModal');
  modal.classList.remove('active');
  document.body.classList.remove('modal-open');
}

// Initialize modal system
document.addEventListener('DOMContentLoaded', () => {
  // Create modal
  createModalHTML();
  
  // Close button click
  document.getElementById('modalClose').addEventListener('click', closeProjectModal);
  
  // Click outside modal to close
  document.getElementById('projectModal').addEventListener('click', (e) => {
    if (e.target.id === 'projectModal') {
      closeProjectModal();
    }
  });
  
  // ESC key to close
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeProjectModal();
      
      // Also close mobile menu if open
      if (navMenu.classList.contains('show')) {
        navMenu.classList.remove('show');
        document.body.classList.remove('menu-open');
        menuToggle.classList.remove('fa-xmark');
        menuToggle.classList.add('fa-bars');
      }
    }
  });
  
  // Attach click handlers to "Learn More" buttons
  document.querySelectorAll('[data-project]').forEach(button => {
    button.addEventListener('click', (e) => {
      e.preventDefault();
      const projectId = button.getAttribute('data-project');
      openProjectModal(projectId);
    });
  });
  
  // Initialize scroll reveal animations
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('fade-in-up');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  const animateElements = document.querySelectorAll(
    '.project-card, .timeline-item, .cert-card, .contact-item, .skill-category'
  );
  
  animateElements.forEach(el => observer.observe(el));
  
  // Initialize active nav link
  updateActiveNavLink();
});

// === Performance: Debounce Scroll Events ===
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

const debouncedScroll = debounce(() => {
  updateActiveNavLink();
}, 10);

window.addEventListener('scroll', debouncedScroll);

// === Log for debugging ===
console.log('Portfolio with modal system loaded successfully! 🚀');
console.log('Projects available:', Object.keys(projectsData));