import { useEffect, useState } from 'react';
import { useNavbarScrolled, useActiveSection } from '../hooks/useActiveSection';

const NAV_IDS = ['hero', 'about', 'projects', 'experience', 'contact'];

const links: { href: string; label: string }[] = [
  { href: '#hero', label: 'Home' },
  { href: '#about', label: 'About' },
  { href: '#projects', label: 'Projects' },
  { href: '#experience', label: 'Experience' },
  { href: '#contact', label: 'Contact' },
];

export function Navbar() {
  const scrolled = useNavbarScrolled();
  const active = useActiveSection(NAV_IDS);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    if (menuOpen) document.body.classList.add('menu-open');
    else document.body.classList.remove('menu-open');
    return () => document.body.classList.remove('menu-open');
  }, [menuOpen]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      const nav = document.getElementById('nav-menu');
      const toggle = document.querySelector('.menu-toggle');
      if (
        menuOpen &&
        nav &&
        toggle &&
        !nav.contains(e.target as Node) &&
        !toggle.contains(e.target as Node)
      ) {
        setMenuOpen(false);
      }
    };
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, [menuOpen]);

  return (
    <nav id="navbar" className={scrolled ? 'scrolled' : undefined}>
      <div className="nav-content">
        <div className="logo">
          Dang<span>.</span>
        </div>
        <ul id="nav-menu" className={menuOpen ? 'show' : undefined}>
          {links.map(({ href, label }) => {
            const id = href.slice(1);
            return (
              <li key={href}>
                <a
                  href={href}
                  className={`nav-link ${active === id ? 'active' : ''}`}
                  onClick={() => setMenuOpen(false)}
                >
                  {label}
                </a>
              </li>
            );
          })}
        </ul>
        <i
          className={`fa-solid ${menuOpen ? 'fa-xmark' : 'fa-bars'} menu-toggle`}
          onClick={() => setMenuOpen((o) => !o)}
          role="button"
          tabIndex={0}
          aria-label="Toggle menu"
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') setMenuOpen((o) => !o);
          }}
        />
      </div>
    </nav>
  );
}
