import { useEffect, useRef, useState } from 'react';

function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function useCountUp(end: number, duration = 1500): { ref: React.RefObject<HTMLElement | null>; count: number } {
  const ref = useRef<HTMLElement | null>(null);
  const [count, setCount] = useState(prefersReducedMotion() ? end : 0);
  const triggered = useRef(false);

  useEffect(() => {
    if (prefersReducedMotion()) {
      setCount(end);
      return;
    }

    const el = ref.current;
    if (!el) return;

    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !triggered.current) {
            triggered.current = true;
            obs.unobserve(entry.target);

            const startTime = performance.now();

            function tick(now: number) {
              const elapsed = now - startTime;
              const progress = Math.min(elapsed / duration, 1);
              // ease-out cubic
              const eased = 1 - Math.pow(1 - progress, 3);
              setCount(Math.round(end * eased));
              if (progress < 1) {
                requestAnimationFrame(tick);
              }
            }

            requestAnimationFrame(tick);
          }
        });
      },
      { threshold: 0.2 }
    );

    obs.observe(el);
    return () => obs.disconnect();
  }, [end, duration]);

  return { ref, count };
}
