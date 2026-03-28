import { useEffect, useState } from 'react';

export function useActiveSection(sectionIds: string[]) {
  const [activeId, setActiveId] = useState(sectionIds[0] ?? 'hero');

  useEffect(() => {
    const onScroll = () => {
      const scrollY = window.scrollY;
      for (let i = sectionIds.length - 1; i >= 0; i--) {
        const id = sectionIds[i];
        const el = document.getElementById(id);
        if (!el) continue;
        const top = el.offsetTop - 120;
        if (scrollY >= top) {
          setActiveId(id);
          return;
        }
      }
      setActiveId(sectionIds[0] ?? 'hero');
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener('scroll', onScroll);
  }, [sectionIds]);

  return activeId;
}

export function useNavbarScrolled(threshold = 50) {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > threshold);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener('scroll', onScroll);
  }, [threshold]);
  return scrolled;
}
